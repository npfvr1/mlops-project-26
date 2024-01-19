"""Tests are commented out because they require the data to be downloaded and processed first."""

# from src.data.make_dataset import *

# def get_expected_lengths():
    
#     expected_len_test__dataset = 2625
#     expected_len_train__dataset = 84635
#     expected_len_val__dataset = 2625
    
#     return [expected_len_test__dataset, expected_len_train__dataset, expected_len_val__dataset]

# # Test that the data is correctly loaded
# def test_loading_data():
    
#     transform = get_transform()
#     datasets = get_data_sets(transform)
#     train_dataset, val_dataset, test_dataset = datasets
    
#     expected_lengths = get_expected_lengths()
#     assert len(test_dataset) ==  expected_lengths[0], "The train dataset is loaded incorrectly"
#     assert len(train_dataset) ==  expected_lengths[1], "The validation dataset is loaded incorrectly"
#     assert len(val_dataset) ==  expected_lengths[2], "The test dataset is loaded incorrectly"
    
# # Verify that the datasets are correctly loaded without any corrupted or missing files.    
# def test_dataset_integrity():
#     datasets = get_data_sets(get_transform())
#     for dataset in datasets:
#         try:
#             for _ in dataset:
#                 pass
#         except Exception as e:
#             assert False, f"Dataset integrity check failed: {e}"
 
# # Test that the labels are consistent across datasets            
# def test_label_consistency():
#     datasets = get_data_sets(get_transform())
#     label_sets = [set() for _ in datasets]

#     for i, dataset in enumerate(datasets):
#         for _, label in dataset:
#             label_sets[i].add(label)

#     assert label_sets[0] == label_sets[1] == label_sets[2], "Labels are not consistent across datasets."

                
# # Run the test
# if __name__ == '__main__':
#     test_loading_data()