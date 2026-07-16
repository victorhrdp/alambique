from alambique.notifications import is_waiting_pass_error


class TestIsWaitingPassError:
    def test_timeout_message(self):
        assert is_waiting_pass_error("pass no respondió en 120s — pinentry")

    def test_other_pass_error(self):
        assert not is_waiting_pass_error("pass no está instalado")

    def test_none(self):
        assert not is_waiting_pass_error(None)