"""Validate logical trial limits before mutating the persisted budget."""
import copy

LIMITS = {'training_starts': 6000, 'evaluation_starts': 144}


def claim_start(budget, key, entry):
    limit = LIMITS[key]
    count = budget[key]
    if not isinstance(count, int) or not 0 <= count < limit:
        raise RuntimeError(f'{key}: {count} already claimed; limit {limit}')
    budget[key] = count + 1
    budget['entries'].append(entry)


def boundary_check():
    checked = []
    for key, count in [('training_starts', 3000), ('training_starts', 5999),
                       ('evaluation_starts', 143)]:
        budget = {key: count, 'entries': []}
        claim_start(budget, key, {'fixture': True})
        assert budget[key] == count + 1 and len(budget['entries']) == 1
        checked.append([key, count, 'accepted'])
    for key, limit in LIMITS.items():
        budget = {key: limit, 'entries': []}
        before = copy.deepcopy(budget)
        try:
            claim_start(budget, key, {'fixture': True})
        except RuntimeError:
            assert budget == before
            checked.append([key, limit, 'rejected_without_mutation'])
        else:
            raise AssertionError(f'{key} failed to reject excess trial')
    return checked
