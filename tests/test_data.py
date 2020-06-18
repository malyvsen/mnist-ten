def test_train_labeled():
    from mnist_ten.data import num_classes
    from mnist_ten.data.mnist import train_labeled
    assert len(train_labeled) == num_classes
    for class_id in range(num_classes):
        assert class_id in [label for image, label in train_labeled]
