"""Run from the project folder with:  python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game import GOAL_H, MALLET_R, PUCK_R, TABLE_H, TABLE_W, Game  # noqa: E402
from vision import SkinTracker, Smoother, TableView  # noqa: E402


def playing_game(**kw) -> Game:
    g = Game(seed=0, **kw)
    g._launch()  # skip the serve delay
    g.puck.vel = np.zeros(2)
    return g


class GameTests(unittest.TestCase):
    def test_puck_bounces_off_top_wall(self):
        g = playing_game()
        g.puck.pos = np.array([400.0, 40.0])
        g.puck.vel = np.array([0.0, -500.0])
        for _ in range(20):
            g.update(1 / 30)
        self.assertGreaterEqual(g.puck.pos[1], PUCK_R - 1e-6)
        self.assertGreater(g.puck.vel[1], 0)  # now moving away from the wall

    def test_puck_bounces_off_side_wall_outside_goal(self):
        g = playing_game()
        g.puck.pos = np.array([40.0, 40.0])  # well above the goal opening
        g.puck.vel = np.array([-500.0, 0.0])
        for _ in range(10):
            g.update(1 / 30)
        self.assertGreaterEqual(g.puck.pos[0], PUCK_R - 1e-6)
        self.assertEqual(g.score, [0, 0])

    def test_goal_for_player(self):
        g = playing_game()
        g.ai.pos = np.array([TABLE_W - MALLET_R, 40.0])  # keep the keeper out of the way
        g.puck.pos = np.array([TABLE_W - 60.0, TABLE_H / 2])
        g.puck.vel = np.array([900.0, 0.0])
        for _ in range(30):
            g.update(1 / 30)
            if g.score[0]:
                break
        self.assertEqual(g.score, [1, 0])
        self.assertEqual(g.state, "serve")  # puck reset to the centre

    def test_goal_for_computer(self):
        g = playing_game()
        g.puck.pos = np.array([60.0, TABLE_H / 2])
        g.puck.vel = np.array([-900.0, 0.0])
        g.set_player_target(300, 40)
        for _ in range(30):
            g.update(1 / 30)
            if g.score[1]:
                break
        self.assertEqual(g.score, [0, 1])

    def test_mallet_hits_puck_away(self):
        g = playing_game()
        g.player.pos = np.array([200.0, 200.0])
        g._target = g.player.pos.copy()
        g.puck.pos = np.array([200.0 + MALLET_R + PUCK_R + 2, 200.0])
        g.set_player_target(260, 200)  # swing the mallet into the puck
        for _ in range(6):
            g.update(1 / 30)
        self.assertGreater(g.puck.vel[0], 100)

    def test_player_mallet_cannot_cross_centre_line(self):
        g = playing_game()
        g.set_player_target(TABLE_W, TABLE_H / 2)
        for _ in range(5):
            g.update(1 / 30)
        self.assertLessEqual(g.player.pos[0], TABLE_W / 2 - MALLET_R + 1e-6)

    def test_fast_puck_does_not_tunnel_through_mallet(self):
        g = playing_game()
        g.player.pos = np.array([300.0, 200.0])
        g._target = g.player.pos.copy()
        g.puck.pos = np.array([420.0, 200.0])
        g.puck.vel = np.array([-1400.0, 0.0])
        for _ in range(10):
            g.update(1 / 30)
        self.assertGreater(g.puck.pos[0], g.player.pos[0])

    def test_match_ends_at_win_score(self):
        g = playing_game(win_score=2)
        for _ in range(2):
            g._score(1)
        self.assertEqual(g.state, "over")
        self.assertEqual(g.winner, "player")
        g.reset_match()
        self.assertEqual(g.score, [0, 0])
        self.assertIsNone(g.winner)

    def test_ai_defends_and_scores_sometimes(self):
        # Long scripted rally: nothing may crash and the puck must stay on the table
        g = Game(difficulty=3, seed=3)
        for k in range(1800):
            g.set_player_target(150 + 100 * np.sin(k / 15), 200 + 120 * np.sin(k / 9))
            g.update(1 / 30)
            self.assertTrue(-PUCK_R * 2 <= g.puck.pos[0] <= TABLE_W + PUCK_R * 2)
            self.assertTrue(0 <= g.puck.pos[1] <= TABLE_H)
        self.assertGreater(sum(g.score), 0)

    def test_goal_opening_is_centred(self):
        self.assertLess(GOAL_H, TABLE_H)


class VisionTests(unittest.TestCase):
    def test_homography_round_trip(self):
        for persp in (True, False):
            v = TableView(960, 540, perspective=persp)
            for x, y in [(0, 0), (800, 400), (400, 200), (123, 321)]:
                fx, fy = v.table_to_frame(x, y)
                tx, ty = v.frame_to_table(fx, fy)
                self.assertAlmostEqual(tx, x, places=2)
                self.assertAlmostEqual(ty, y, places=2)

    def test_perspective_table_is_narrower_at_the_top(self):
        v = TableView(960, 540, perspective=True)
        top = v.table_to_frame(800, 0)[0] - v.table_to_frame(0, 0)[0]
        bottom = v.table_to_frame(800, 400)[0] - v.table_to_frame(0, 400)[0]
        self.assertLess(top, bottom)

    def test_overlay_keeps_frame_shape_and_changes_pixels(self):
        v = TableView(960, 540)
        frame = np.zeros((540, 960, 3), np.uint8)
        out = v.overlay(frame, np.full((TABLE_H, TABLE_W, 3), 200, np.uint8))
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(out[300, 480].sum(), 0)   # inside the table
        self.assertEqual(out[5, 5].sum(), 0)         # outside the table

    def test_smoother_reduces_jitter(self):
        s = Smoother(alpha=0.5)
        s.update(0, 0)
        x, _ = s.update(100, 0)
        self.assertAlmostEqual(x, 50.0)

    def test_skin_tracker_finds_fingertip(self):
        frame = np.full((540, 960, 3), (60, 60, 60), np.uint8)
        skin = (120, 160, 215)  # BGR, inside the YCrCb skin range
        probe = cv2.cvtColor(np.uint8([[skin]]), cv2.COLOR_BGR2YCrCb)[0, 0]
        self.assertTrue(133 <= probe[1] <= 173 and 77 <= probe[2] <= 127, probe)
        cv2.ellipse(frame, (500, 420), (90, 80), 0, 0, 360, skin, -1)      # palm
        cv2.rectangle(frame, (475, 250), (525, 400), skin, -1)             # index finger
        tracker = SkinTracker()
        tip = tracker.process(frame)
        self.assertIsNotNone(tip)
        self.assertAlmostEqual(tip[0], 500, delta=15)
        self.assertAlmostEqual(tip[1], 250, delta=10)
        self.assertEqual(tracker.mask.shape, frame.shape[:2])
        tracker.draw(frame)  # must not raise

    def test_skin_tracker_returns_none_without_hand(self):
        frame = np.full((540, 960, 3), (60, 60, 60), np.uint8)
        self.assertIsNone(SkinTracker().process(frame))


if __name__ == "__main__":
    unittest.main()
