#include <cstdio>
#include <string>
#include <string_view>

#include <limits.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct EarlyExit {
    const char *argument;            /**< The one argument passed to the program. */
    int expectedExitCode;            /**< The code the program must exit with. */
    std::string_view expectedOutput; /**< Text its standard output must hold once read through a pipe. */
};

struct Outcome {
    bool observed;              /**< False when the child could not be started or waited for. */
    int waitStatus;             /**< The status `waitpid` returned for the child. */
    std::string standardOutput; /**< Everything the child wrote to the pipe. */
};

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

[[nodiscard]] std::string directoryOfThisProgram()
{
    char self[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (length <= 0)
        return {};
    const std::string path{self, static_cast<std::size_t>(length)};
    return path.substr(0, path.find_last_of('/'));
}

[[nodiscard]] std::string readUntilClosed(int descriptor)
{
    std::string received;
    char buffer[512];
    for (ssize_t count = read(descriptor, buffer, sizeof(buffer)); count > 0;
         count = read(descriptor, buffer, sizeof(buffer)))
        received.append(buffer, static_cast<std::size_t>(count));
    return received;
}

[[noreturn]] void replaceThisChildWithProgram(int pipeWriteEnd, int pipeReadEnd, const std::string &program,
                                              const char *argument)
{
    dup2(pipeWriteEnd, STDOUT_FILENO);
    close(pipeWriteEnd);
    close(pipeReadEnd);
    execl(program.c_str(), program.c_str(), argument, static_cast<char *>(nullptr));
    std::perror(program.c_str());
    _exit(127);
}

[[nodiscard]] Outcome runWithStandardOutputPiped(const std::string &program, const char *argument)
{
    int pipeEnds[2];
    if (pipe(pipeEnds) != 0)
        return {.observed = false, .waitStatus = 0, .standardOutput = {}};
    const pid_t child = fork();
    if (child < 0)
    {
        close(pipeEnds[0]);
        close(pipeEnds[1]);
        return {.observed = false, .waitStatus = 0, .standardOutput = {}};
    }
    if (child == 0)
        replaceThisChildWithProgram(pipeEnds[1], pipeEnds[0], program, argument);
    close(pipeEnds[1]);
    Outcome outcome{.observed = true, .waitStatus = 0, .standardOutput = readUntilClosed(pipeEnds[0])};
    close(pipeEnds[0]);
    outcome.observed = waitpid(child, &outcome.waitStatus, 0) == child;
    return outcome;
}

void printHowTheProcessEnded(const Outcome &outcome)
{
    if (!outcome.observed)
        std::printf("    the child process could not be started or waited for\n");
    else if (WIFSIGNALED(outcome.waitStatus))
        std::printf("    killed by signal %d\n", WTERMSIG(outcome.waitStatus));
    else
        std::printf("    exit code %d\n", WEXITSTATUS(outcome.waitStatus));
    std::printf("    %zu bytes reached the pipe\n", outcome.standardOutput.size());
}

void checkEarlyExit(const std::string &program, const EarlyExit &expected)
{
    std::printf("── lpl-assistant %s\n", expected.argument);
    const Outcome outcome = runWithStandardOutputPiped(program, expected.argument);
    printHowTheProcessEnded(outcome);
    const bool exitedOnItsOwn = outcome.observed && WIFEXITED(outcome.waitStatus);
    check(exitedOnItsOwn, "the process exits rather than being killed");
    check(exitedOnItsOwn && WEXITSTATUS(outcome.waitStatus) == expected.expectedExitCode,
          "the exit code is the one main returned");
    check(outcome.standardOutput.find(expected.expectedOutput) != std::string::npos,
          "what main printed reaches the pipe");
}

} // namespace

/**
 * @brief Runs lpl-assistant, from the same build directory, once per way it leaves before loading a
 * model, with its standard output on a pipe.
 *
 * @note Each of these returns from main, so each runs every destructor registered at exit. That is
 * where a destructor run twice aborts the process (Christian-guajardo/LplAssistant#82), and a
 * pipe is where an abort shows twice: the exit code, and the buffered output that never arrives.
 *
 * @return 0 when every check passes, 1 otherwise.
 */
int main()
{
    std::printf("test-early-exit — the exits that come before any model\n");

    const std::string directory = directoryOfThisProgram();
    check(!directory.empty(), "the build directory is found from /proc/self/exe");
    const std::string program = directory + "/lpl-assistant";

    constexpr EarlyExit kEarlyExits[] = {
        {.argument = "--help",         .expectedExitCode = 0, .expectedOutput = "Usage: "        },
        {.argument = "--version",      .expectedExitCode = 0, .expectedOutput = "lpl-assistant ("},
        {.argument = "--no-such-flag", .expectedExitCode = 1, .expectedOutput = "Usage: "        },
    };
    for (const EarlyExit &earlyExit : kEarlyExits)
        checkEarlyExit(program, earlyExit);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
