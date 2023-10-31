import unittest

import numpy as np
from ReversiHelpers import *

class TestOthelloEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.env = OthelloEnvironment()
    
    def setUp(self):
        self.env.reset()

    def tearDown(self):
        self.env.reset()
        
    def test_init(self):
        self.assertTrue(
            np.array_equal(self.env.board,
                           np.array([   [ 0,  0,  0,  0,  0,  0,  0,  0],
                                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                                        [ 0,  0,  0,  1, -1,  0,  0,  0],
                                        [ 0,  0,  0, -1,  1,  0,  0,  0],
                                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                                        [ 0,  0,  0,  0,  0,  0,  0,  0]])))

    def test_init_action(self):
        self.assertTrue(
            np.array_equal(
                self.env.get_legal_moves(return_as="list"), 
                np.array([  [2, 3],
                            [3, 2],
                            [4, 5],
                            [5, 4]
                        ])
            ))
    
    def test_init_current_player(self):
        self.assertEqual(self.env.current_player, DISK_BLACK)

    def test_good_step(self):
        (board, player), reward, terminated, truncated, info=self.env.step((2,3))
        self.assertTrue(np.array_equal(
            board, 
            np.array([  [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0, -1,  0,  0,  0,  0],
                        [ 0,  0,  0, -1, -1,  0,  0,  0],
                        [ 0,  0,  0, -1,  1,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0]])))

        self.assertEqual(player, 1) # White to move
        self.assertEqual(reward, 0) # No win, no reward
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue('legal_moves' in info.keys())
        self.assertTrue(
            np.array_equal(
                            info['legal_moves'], 
                            np.array([  [2, 2],
                                        [2, 4],
                                        [4, 2]])
                        ))
    
    def test_bad_step(self):
        (board, player), reward, terminated, truncated, info=self.env.step((-1,-1))
        self.assertTrue(np.array_equal(
            board, 
            np.array([  [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  1, -1,  0,  0,  0],
                        [ 0,  0,  0, -1,  1,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0]])))

        self.assertEqual(player, 0) # Black to move
        self.assertEqual(reward, 0) # No win, no reward
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {'error':'invalid action'})

    def test_loss_step(self):
        init_state = np.array([     [ 0, -1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1]])
        self.env.set_state(init_state, DISK_WHITE)
        (board, player), reward, terminated, truncated, info=self.env.step((0,0))
        self.assertTrue(np.array_equal(
            board, 
            np.ones((8,8))))

        self.assertEqual(player, 1) # White to move
        self.assertEqual(reward, -1) # No win, no reward
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_win_step(self):
        init_state = np.array([     [ 0, -1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1]])
        self.env.set_state(init_state, DISK_WHITE, DISK_WHITE)
        (board, player), reward, terminated, truncated, info=self.env.step((0,0))
        self.assertTrue(np.array_equal(
            board, 
            np.ones((8,8))))

        self.assertEqual(player, 1) # White to move
        self.assertEqual(reward, 1) # Win
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
    
    def test_draw_step(self):
        init_state = np.array([     [ 0, -1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1]])
        self.env.set_state(init_state, DISK_WHITE)
        (board, player), reward, terminated, truncated, info=self.env.step((0,0))
        self.assertTrue(np.array_equal(
            board, 
            np.array([  [ 1,  1,  1,  1,  1,  1,  1,  1],
                        [ 1,  1,  1,  1,  1,  1,  1,  1],
                        [ 1,  1,  1,  1,  1,  1,  1,  1],
                        [ 1,  1,  1,  1,  1,  1,  1,  1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1]])
        ))
        self.assertEqual(player, 1) # White to move
        self.assertEqual(reward, -1) # No win, no reward
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
        
        
    def test_mockGame(self):
        for game in range(20): # change count to change number of mock games played
            self.env.reset()
            game_ended = False
            legal_moves = self.env.get_legal_moves(return_as="list")
            board,reward = None,None
            for i in range(64):
                if(len(legal_moves) == 0):
                    break
                (board, _), reward, terminated, _, info= self.env.step( legal_moves[np.random.choice(len(legal_moves))])
                if('error' in info.keys()):
                    self.fail("Should not have chosen a bad move")
                if(terminated):
                    game_ended = True
                    break
                legal_moves = info['legal_moves']  
            self.assertTrue(game_ended)

if __name__ == '__main__':
    unittest.main()