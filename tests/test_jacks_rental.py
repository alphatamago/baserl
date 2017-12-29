import unittest

from baserl.jacks_rental import JacksRental

class JacksRentalTestCase(unittest.TestCase):
    def test_move_cars(self):
        max_cars = 10
        jacks_rental = JacksRental(max_cars=max_cars)
        self.assertEqual(jacks_rental.move_cars_((0,0), 1), ((0,0), 0))
        self.assertEqual(jacks_rental.move_cars_((0,0), -1), ((0,0), 0))
        self.assertEqual(jacks_rental.move_cars_((3,4), 1), ((2,5), 1))
        self.assertEqual(jacks_rental.move_cars_((3,4), -1), ((4,3), -1))
        self.assertEqual(jacks_rental.move_cars_((max_cars-3,max_cars-1), 5), ((max_cars-8,max_cars), 5))
        self.assertEqual(jacks_rental.move_cars_((max_cars-3,max_cars), -5), ((max_cars,max_cars-5), -5))

    def test_adjust_state(self):
        jacks_rental = JacksRental(max_cars=20)
        self.assertEqual(jacks_rental.adjust_state_(0, 0, 0), 0)
        self.assertEqual(jacks_rental.adjust_state_(1, 1, 0), 0)
        self.assertEqual(jacks_rental.adjust_state_(10, 1, 5), 14)
        self.assertEqual(jacks_rental.adjust_state_(1, 1, 5), 5)
        
if __name__ == '__main__':
    unittest.main()
