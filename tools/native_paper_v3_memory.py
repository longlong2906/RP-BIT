"""Absolute RAM admission for v3 validation.

Promoted from the successfully used 2026-11-03 frozen-v2 resource-only repair.
Credits only conservatively clean inactive file pages at every cgroup ancestor.
Does not change cgroup limits, swap, global cache, geometry or trial results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys

VERSION = 'posthoc_clean_inactive_file_admission_v1'
GIB = 1024**3
REFERENCE_URLS = [
    'https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html',
    'https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html',
    'https://docs.docker.com/reference/cli/docker/container/stats/',
]


class AdmissionError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def integer(value, label, *, positive=False):
    if isinstance(value, bool):
        raise AdmissionError(label + ': boolean is not a byte counter')
    if isinstance(value, str):
        if not re.fullmatch(r'[0-9]+', value.strip()):
            raise AdmissionError(label + ': malformed counter')
        value = int(value.strip())
    if not isinstance(value, int) or value < (1 if positive else 0):
        raise AdmissionError(label + ': expected a nonnegative integer')
    return value


def read_text(path):
    return Path(path).read_text(encoding='ascii').strip()


def parse_stat(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in result:
            raise AdmissionError('malformed or duplicate memory.stat row')
        result[fields[0]] = integer(fields[1], 'memory.stat ' + fields[0])
    return result


def cache_credit(stat, current, version, hierarchy=None):
    """Credit only conservatively clean inactive file-backed pages.

    Dirty/writeback bytes are subtracted even when outside the inactive list;
    this can under-credit, but never intentionally credits those busy pages.
    Active file cache, swap, slab, anon and shmem are not added as headroom.
    """
    if version == 2:
        keys = ('inactive_file', 'file', 'shmem', 'file_dirty', 'file_writeback')
        if any(key not in stat for key in keys):
            return 0, 'missing_v2_credit_fields'
        inactive, file_bytes, shmem, dirty, writeback = (stat[k] for k in keys)
        if shmem > file_bytes:
            return 0, 'inconsistent_file_shmem_counters'
        cap = min(inactive, file_bytes - shmem, current)
    elif version == 1:
        if hierarchy not in (0, 1):
            return 0, 'unknown_v1_hierarchy_scope'
        prefix = 'total_' if hierarchy == 1 else ''
        keys = tuple(prefix + k for k in ('inactive_file', 'cache', 'dirty', 'writeback'))
        if any(key not in stat for key in keys):
            return 0, 'missing_v1_credit_fields'
        inactive, file_bytes, dirty, writeback = (stat[k] for k in keys)
        shmem = stat.get(prefix + 'shmem')
        if shmem is not None:
            if shmem > file_bytes:
                return 0, 'inconsistent_cache_shmem_counters'
            file_bytes -= shmem
        # v1 inactive_file is a file-backed LRU (plus reclaimable MADV_FREE
        # pages); shmem is on anon LRUs. Never substitute total cache for it.
        cap = min(inactive, file_bytes, current)
    else:
        return 0, 'unknown_cgroup_version'
    return max(0, cap - dirty - writeback), 'clean_inactive_file_only'


def sample_source(source, reader=read_text):
    detected_limit = integer(source['limit_bytes'], 'cgroup limit', positive=True)
    current_path = Path(source['current_file'])
    if current_path.name == 'memory.current':
        version = 2
    elif current_path.name == 'memory.usage_in_bytes':
        version = 1
    else:
        raise AdmissionError('unrecognized constrained memory current file')
    directory = current_path.parent
    limit_path = directory / ('memory.max' if version == 2 else 'memory.limit_in_bytes')
    live_limit_text = reader(limit_path).strip()
    live_limit = None if version == 2 and live_limit_text == 'max' else integer(live_limit_text, str(limit_path), positive=True)
    limit = detected_limit if live_limit is None else min(detected_limit, live_limit)
    hierarchy = None
    if version == 1:
        try:
            hierarchy = integer(reader(directory / 'memory.use_hierarchy'), 'memory.use_hierarchy')
            if hierarchy not in (0, 1):
                hierarchy = None
        except (OSError, UnicodeError, AdmissionError):
            hierarchy = None
    current = integer(reader(current_path), str(current_path))
    try:
        stats = parse_stat(reader(directory / 'memory.stat'))
        credit, reason = cache_credit(stats, current, version, hierarchy)
        stat_error = None
    except (OSError, UnicodeError, AdmissionError) as error:
        stats, credit, reason, stat_error = {}, 0, 'unreadable_or_malformed_stat_no_credit', repr(error)
    return {'version': version, 'current_file': str(current_path), 'limit_bytes': limit,
            'detected_limit_bytes': detected_limit, 'live_limit_text': live_limit_text,
            'current_bytes': current, 'hierarchy': hierarchy, 'memory_stat': stats,
            'clean_inactive_credit_bytes': credit, 'credit_reason': reason, 'stat_error': stat_error}


def host_sample(host_reader):
    host = host_reader()
    total = integer(host['total'], 'host total', positive=True)
    available = integer(host['available'], 'host available')
    if available > total:
        raise AdmissionError('host available exceeds total')
    return {'total': total, 'available': available}


def admission_snapshot(runtime, host_reader, reader=read_text):
    """Use absolute headroom at every constrained ancestor, never mixed ratios."""
    evidence = {'policy_version': VERSION, 'sources': [], 'credit_is_an_estimate_not_an_allocation_guarantee': True}
    try:
        limit = integer(runtime['memory_limit_bytes'], 'runtime memory limit', positive=True)
        before = host_sample(host_reader)
        evidence['host_before'] = before
        for source in runtime['memory_sources']:
            first = sample_source(source, reader)
            second = sample_source(source, reader)
            # Counters are not an atomic transaction. Charge the larger observed
            # usage and credit only the smaller independently observed clean LRU.
            current = max(first['current_bytes'], second['current_bytes'])
            credit = min(first['clean_inactive_credit_bytes'], second['clean_inactive_credit_bytes'],
                         first['current_bytes'], second['current_bytes'])
            if first['hierarchy'] != second['hierarchy']:
                credit = 0
            effective_usage = max(0, current - credit)
            effective_limit = min(first['limit_bytes'], second['limit_bytes'])
            headroom = max(0, effective_limit - effective_usage)
            evidence['sources'].append({'first': first, 'second': second, 'charged_current_bytes': current,
                                        'credited_clean_inactive_bytes': credit,
                                        'effective_limit_bytes': effective_limit,
                                        'effective_usage_bytes': effective_usage, 'headroom_bytes': headroom})
        after = host_sample(host_reader)
        evidence['host_after'] = after
        available = min(limit, before['available'], after['available'],
                        *(s['headroom_bytes'] for s in evidence['sources']))
        pressure = min(1.0, max(0.0, 1.0 - available / limit))
        if limit * (1.0 - pressure) > available:
            pressure = math.nextafter(pressure, 1.0)
        legacy_pressure = max((before['total'] - before['available']) / before['total'],
                              (after['total'] - after['available']) / after['total'],
                              *(s['charged_current_bytes'] / s['first']['detected_limit_bytes'] for s in evidence['sources']))
        effective_available = limit * (1.0 - pressure)
        evidence.update(runtime_memory_limit_bytes=limit, absolute_headroom_bytes=available,
                        returned_pressure=pressure, frozen_validator_available_bytes=effective_available,
                        legacy_available_bytes_from_same_samples=limit * (1.0 - legacy_pressure),
                        per_worker_reservation_bytes=2 * GIB, retained_admission_fraction=.8,
                        admission_would_allow_one_worker=effective_available * .8 >= 2 * GIB,
                        reservation_slots=int(effective_available * .8 // (2 * GIB)))
        return evidence
    except (OSError, UnicodeError, KeyError, AdmissionError) as error:
        evidence['error'] = repr(error)
        raise AdmissionError('POSTHOC_RAM_ADMISSION_UNREADABLE_OR_INVALID: ' + str(error), evidence) from error

