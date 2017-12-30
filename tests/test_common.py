import unittest

from baserl.common import *

class CommonTestCase(unittest.TestCase):
    def test_chop_prob_distribution(self):
        self.assertEqual(len(chop_prob_distribution([('a', .90), ('b', .09),
                                                     ('c', .01)], .90, 1)), 1)
        self.assertEqual(len(chop_prob_distribution([('a', 90), ('b', 9),
                                                     ('c', 1)], .90, 1)), 1)
        self.assertEqual(len(chop_prob_distribution([('a', .90), ('b', .09),
                                                     ('c', .01)], .99, 1)), 2)
        self.assertEqual(len(chop_prob_distribution([('a', 90), ('b', 9),
                                                     ('c', 1)], .99, 1)), 2)
        self.assertEqual(len(chop_prob_distribution([('a', .90), ('b', .09),
                                                     ('c', .01)], .50, 1)), 1)
        self.assertEqual(len(chop_prob_distribution([('a', .90), ('b', .09),
                                                     ('c', .01)], .999, 1)), 3)
        self.assertEqual(len(chop_prob_distribution([('a', .90), ('b', .09),
                                                     ('c', .01)], 1.0, 1)), 3)
        self.assertEqual(len(chop_prob_distribution([('a', .9), ('b', .9),
                                                     ('c', .9)], 1.0, 1)), 3)
        self.assertEqual(len(chop_prob_distribution([('a', .0001), ('b', .0001),
                                                     ('c', .0001)], 1.0, 1)), 3)
