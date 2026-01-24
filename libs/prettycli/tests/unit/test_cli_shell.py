"""测试 CLI 的 bash 会话持久化"""
from prettycli.cli import CLI


class TestCLIShell:
    """测试 CLI shell 会话"""

    def test_shell_session_created_on_demand(self):
        """shell 会话按需创建"""
        cli = CLI("test")
        assert cli._shell is None

        shell = cli._get_shell()
        assert shell is not None
        assert shell.is_alive

        shell.close()

    def test_shell_session_reused(self):
        """shell 会话复用"""
        cli = CLI("test")

        shell1 = cli._get_shell()
        shell2 = cli._get_shell()
        assert shell1 is shell2

        shell1.close()

    def test_shell_cd_persists(self):
        """cd 命令状态持久化"""
        cli = CLI("test")

        cli._run_bash("cd /tmp")
        result = cli._get_shell().run("pwd")
        assert result.stdout == "/tmp"

        cli._shell.close()

    def test_shell_env_persists(self):
        """环境变量持久化"""
        cli = CLI("test")

        cli._run_bash("export TEST_VAR=hello123")
        result = cli._get_shell().run("echo $TEST_VAR")
        assert result.stdout == "hello123"

        cli._shell.close()

    def test_shell_exit_code(self):
        """返回正确的退出码"""
        cli = CLI("test")

        code = cli._run_bash("true")
        assert code == 0

        code = cli._run_bash("false")
        assert code != 0

        cli._shell.close()

    def test_shell_output_captured(self):
        """输出被正确捕获"""
        cli = CLI("test")

        cli._run_bash("echo hello")
        assert "hello" in cli._last_output

        cli._shell.close()

    def test_shell_recreated_if_dead(self):
        """shell 关闭后自动重建"""
        cli = CLI("test")

        shell1 = cli._get_shell()
        shell1.close()

        shell2 = cli._get_shell()
        assert shell2 is not shell1
        assert shell2.is_alive

        shell2.close()
