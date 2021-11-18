class Range:

    def __init__(self, left, right, left_inclusive=True, right_inclusive=False):
        self.left = left
        self.right = right
        self.left_inclusive = left_inclusive
        self.right_inclusive = right_inclusive

    def __contains__(self, item):

        if self.left_inclusive and self.left == item:
            return True
        if self.right_inclusive and self.right == item:
            return True

        return self.left < item < self.right


class OpenRange(Range):

    def __init__(self, left, right):
        super().__init__(left, right, False, False)


class ClosedRange(Range):

    def __init__(self, left, right):
        super().__init__(left, right, True, True)
