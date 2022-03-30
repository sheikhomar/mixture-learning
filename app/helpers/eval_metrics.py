from typing import List


def type_1_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    all_indices = list(range(data_set_size))
    correctly_labelled_indices = list(set(all_indices) - set(known_mislabelled_indices))

    # Find samples which are correctly labelled but are detected as mislabelled.
    type_1_error_indices = [
        index
        for index in detected_mislabelled_indices
        if index in correctly_labelled_indices
    ]

    # Type 1 errors are correctly labelled instances that
    # are erroneously identified as mislabelled.
    er1 = len(type_1_error_indices) / len(correctly_labelled_indices)

    return er1


def type_2_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Type 2 errors are known mislabeled instances which are not detected.
    type_2_indices = [
        index
        for index in known_mislabelled_indices
        if index not in detected_mislabelled_indices
    ]
    if len(known_mislabelled_indices) > 0:
        er2 = len(type_2_indices) / len(known_mislabelled_indices)
    else:
        er2 = 0.0
    return er2


def noise_elimination_precision_score(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Noise elimination precision (NEP) is the percentage of
    # detected instances that are known to be mislabelled.
    detected_and_known_indices = [
        index
        for index in known_mislabelled_indices
        if index in detected_mislabelled_indices
    ]
    nep = len(detected_and_known_indices) / len(detected_mislabelled_indices)
    return nep
