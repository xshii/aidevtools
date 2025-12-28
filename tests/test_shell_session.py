"""Shell 会话测试示例 - 验证状态持久化"""
from pathlib import Path

from prettycli.shell import ShellSession
from tests.framework import test, assert_equals, assert_contains, TestRunner


@test("should persist working directory after cd")
def test_cd_persistence():
    with ShellSession() as sh:
        # 切换到 /tmp
        sh.run("cd /tmp")
        result = sh.run("pwd")
        assert_equals(result.stdout, "/tmp")

        # 再切换到子目录
        sh.run("mkdir -p test_prettycli")
        sh.run("cd test_prettycli")
        result = sh.run("pwd")
        assert_equals(result.stdout, "/tmp/test_prettycli")

        # 清理
        sh.run("cd /tmp && rm -rf test_prettycli")


@test("should persist environment variables")
def test_env_persistence():
    with ShellSession() as sh:
        # 设置环境变量
        sh.run("export MY_VAR=hello")
        result = sh.run("echo $MY_VAR")
        assert_equals(result.stdout, "hello")

        # 修改环境变量
        sh.run("export MY_VAR=world")
        result = sh.run("echo $MY_VAR")
        assert_equals(result.stdout, "world")

        # 使用快捷方法
        sh.export("FOO", "bar")
        assert_equals(sh.env("FOO"), "bar")


@test("should capture exit code")
def test_exit_code():
    with ShellSession() as sh:
        # 成功命令
        result = sh.run("echo ok")
        assert_equals(result.exit_code, 0)

        # 失败命令
        result = sh.run("ls /nonexistent 2>/dev/null")
        assert result.exit_code != 0


@test("should handle multi-line output")
def test_multiline_output():
    with ShellSession() as sh:
        result = sh.run("echo -e 'line1\\nline2\\nline3'")
        lines = result.stdout.split("\n")
        assert_equals(len(lines), 3)
        assert_equals(lines[0], "line1")
        assert_equals(lines[2], "line3")


@test("should support pipes and redirects")
def test_pipes():
    with ShellSession() as sh:
        result = sh.run("echo 'hello world' | grep world")
        assert_contains(result.stdout, "world")

        result = sh.run("echo test > /tmp/prettycli_test.txt && cat /tmp/prettycli_test.txt")
        assert_equals(result.stdout, "test")

        # 清理
        sh.run("rm -f /tmp/prettycli_test.txt")


@test("should track command history")
def test_history():
    with ShellSession() as sh:
        sh.run("echo first")
        sh.run("echo second")
        sh.run("echo third")

        assert_equals(len(sh.history), 3)
        assert_equals(sh.history[0].command, "echo first")
        assert_equals(sh.last_result.command, "echo third")


@test("should use helper methods")
def test_helpers():
    with ShellSession() as sh:
        # 使用 cd 方法
        assert sh.cd("/tmp") is True
        assert_equals(sh.pwd(), "/tmp")

        # cd 到不存在的目录
        assert sh.cd("/nonexistent_dir_12345") is False


if __name__ == "__main__":
    import sys
    runner = TestRunner()
    runner.discover(Path(__file__).parent)
    sys.exit(runner.run())  # 失败时返回非零退出码
