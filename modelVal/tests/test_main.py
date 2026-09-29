import sys
import unittest
from unittest.mock import patch

import main


class MainCliTests(unittest.TestCase):
    def test_default_gui_command_sets_public_to_false(self):
        captured = {}

        def capture_args(args):
            captured["args"] = args

        with patch.object(sys, "argv", ["main.py"]), \
                patch.object(main, "cmd_gui", capture_args):
            main.main()

        self.assertEqual(False, getattr(captured["args"], "public", None))


if __name__ == "__main__":
    unittest.main()
