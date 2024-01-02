# Contributing

### Contents

- [Contributing](#contributing)
    - [Contents](#contents)
    - [Choose an issue to work on](#choose-an-issue-to-work-on)
    - [Pull request checklist](#pull-request-checklist)
  - [Release Notes](#release-notes)
    - [Adding a new release note](#adding-a-new-release-note)
      - [Linking to issues](#linking-to-issues)
    - [Generating the release notes](#generating-the-release-notes)
    - [Building release notes locally](#building-release-notes-locally)
  - [Coding Style](#coding-style)
    - [Debug Output](#debug-output)
    - [Static Code Checks](#static-code-checks)
  - [Testing and Analysis](#testing-and-analysis)
    - [Debugging LIT Tests](#debugging-lit-tests)
    - [Setting Paths for Manual Test Runs](#setting-paths-for-manual-test-runs)
    - [Adding Unit Tests](#adding-unit-tests)
  - [CI and Release Cycle](#ci-and-release-cycle)
    - [Branches](#branches)
    - [Tags](#tags)
    - [Release cycle](#release-cycle)
    - [Example release cycle](#example-release-cycle)
  - [Notes](#notes)
    - [Conan dependencies](#conan-dependencies)
      - [Bumping dependency versions:](#bumping-dependency-versions)


### Choose an issue to work on
The project uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

* [good first issue](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, perfect for newcomers. They should already be fully scoped, with a clear approach outlined in the descriptions.
* [documentation](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug+label%3A%22good+first+issue%22+label%3Aresearch+label%3Adocumentation) - A documentation task. These are often great for newcomers to the project.
* [enhancement](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug+label%3A%22good+first+issue%22+label%3Aresearch+label%3Adocumentation+label%3Aquestion+label%3ATask+label%3Aenhancement) - A new feature or request for feature.
* [question](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug+label%3A%22good+first+issue%22+label%3Aresearch+label%3Adocumentation+label%3Aquestion) - An open question for discussion.
* [bug](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug) - An open bug that should be resolved.
* [research](https://github.com/openqasm/qe-compiler/issues?q=is%3Aopen+is%3Aissue+label%3Abug+label%3A%22good+first+issue%22+label%3Aresearch) - A research oriented issue. A good place to start for more nuanced and open ended issues.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute
   `cmake --build . --target check-all` locally,
   which will run these checks and report any issues.

   If your code fails the local style checks (specifically the black
   code formatting check) you can use `cmake --build . --target fix-format` to automatically
   fix update the code formatting.

2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* (for C++ and Python use Doxygen and Sphinx style docstrings respectively).

3. If it makes sense for your change that you have added new tests that
   cover the changes.

4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for that
   change and that the PR is tagged for the changelog.


## Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of the compiler. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses the compiler, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

### Adding a new release note

Making a new release note is quite straightforward.

- Ensure that you have reno installed with:
    ```bash
    pip install -U reno
    ```

- Once you have reno installed you can make a new release note by running in
your local repository root:
    ```bash
    reno new <short-description-string>
    ```
    where <short-description-string> is a brief string (with no spaces) that describes
    what's in the release note. This will become the prefix for the release note
    file. Once that is run it will create a new yaml file in `releasenotes/notes`.
- Open the created yaml file in a text editor and write the release note. The basic
    structure of a release note is restructured text in yaml lists under category
    keys. You add individual items under each category and they will be grouped
    automatically by release when the release notes are compiled. A single file
    can have as many entries in it as needed, but to avoid potential conflicts
    you'll want to create a new file for each pull request that has user facing
    changes. When you open the newly created file it will be a full template of
    the different categories with a description of a category as a single entry
    in each category. You'll want to delete all the sections you aren't using and
    update the contents for those you are. For example, the end result should
    look something like:

    ```yaml
    features:
    - |
        Introduced a new feature foo, that adds support for doing something

    deprecations:
    - |
        ``Foo`` has been deprecated and will be removed in a
        future release.
    ```

You can also look at other release notes for other examples.

Note that you can use sphinx [restructured text syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
In fact, you can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

#### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/openqasm/qe-compiler/issues/12345>` for more
    details.
```

### Generating the release notes

After release notes have been added, you can use reno to see what the full output
of the release notes is. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full qe-compiler release
notes for all releases (since we started using reno during 0.9) you just run:

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged):

    reno report --version 0.9.0

### Building release notes locally

Building The release notes are part of the standard compiler documentation
builds. See the [README](./README.md) for more information on how to build the documentation.

## Coding Style

We follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) as our overall guideline, with some exceptions noted below. That is, rules stated below take precedence.

- Be kind when you think you need to comment on style in a PR. We follow a common coding style to reduce friction when working together.

- Naming
  - Use descriptive names; type names and variable names should be nouns, function names should be verb phrases.
  - Capitalize names that consist of several words in CamelCase.
    - Type names, enum declarations and enumerators start with an upper-case letter (e.g., `TargetSystem`).
    - Variable names, function names, function parameters, and data members start with a lower-case letter (e.g., `funcMap`).
      - LLVM mandates upper-case yet our code follows lower-case convention.
  - Add a trailing underscore for private members' names.

- Prefer error handling as discussed in the [LLVM's Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html#error-handling).
  - Fail early on programmatic errors, that is violation of invariants or API contracts.
  - Report recoverable errors via `llvm::Error` or `llvm::Expected<T>`.
  - When failing because of user input, provide the user with an actionable error message.

- Prefer using [LLVM's replacement containers and string handling classes](https://llvm.org/docs/ProgrammersManual.html#picking-the-right-data-structure-for-a-task)
  - LLVM's data structure integrate more smoothly with LLVM's APIs and avoid heap allocation.
  - On the flip side, they can require convoluted conversions -- prefer STL containers when they allow writing cleaner and saner code.

- Ownership
  - [Raw pointers and references are always non-owning]((https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-ptr)), use them whenever possible and ownership is handled elsewhere.
  - [Prefer scoped objects over heap allocation](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r5-prefer-scoped-objects-dont-heap-allocate-unnecessarily) when it allows you to stop worrying about ownership. Lifetime is handled together with the containing scope or object.
  - Use smart pointers judiciously when you do need to pass ownership (and that should be rare). [Use a `std::unique_ptr` to represent ownership](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-unique).
  - Keep in mind that `std::shared_ptr` is expensive and only useful with multiple threads. Note that a `shared_ptr` does not prevent data races in the pointed-to objects and you need to take care of them separately.

- Use regular function declarator syntax, not trailing return type declaration.

- File Headers are required
  - Every source and header file must have a header with a copyright remark.
  - The header should describe the basic purpose of the file.
  - See [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#file-headers)
  - Note that the second part of the header is a doxygen file comment and is prefixed with three slashes `///`.

- All PRs must compile and test successfully before merging (i.e., on the last commit), as enforced by the CI.

- Order `#include` statements according to the [LLVM Coding Standard](https://llvm.org/docs/CodingStandards.html#include-style) (with MLIR headers as a separate block), grouped in these blocks with empty lines between blocks:
  1. Main Module Header (i.e., the interface that we implement in a given .cpp)
  2. Local/Private Headers
  3. MLIR project/subproject headers (as separate block)
  4. LLVM project/subproject headers (`clang/...`, `lldb/...`, `llvm/...`, etc, except MLIR)
  5. System #includes

### Debug Output

Note that `llvm::outs()` is not thread-safe (since it is buffered) and thus cannot be used in code that will run multi-threaded. For short-lived debug messages, printing to `llvm:errs()` is fine (also thread-safe). If debug messages should remain in the code, consider using `LLVM_DEBUG` (see [the LLVM Programmer's Reference Manual - Section The LLVM_DEBUG() macro and -debug option](https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option)).

### Static Code Checks
The easiest, fastest, and most automated way to integrate the formatting into your workflow
is via [pre-commit](https://pre-commit.com). Note that this tool requires and internet connection
to initially setup because the formatting tools needs to be downloaded.

**In environments without an internet connection, please see one of the other solutions documented
below.**

These should be installed and setup prior to any development work so that they are not forgotten
about. The setup is straight forward:

```bash
pip install pre-commit
pre-commit install
```

The first time `pre-commit install` is run, it will take a minute to setup the environment.

After installation, the hooks will be run prior to every commit, and will be run against all staged
changes. Optionally, you can trigger this run via `pre-commit run`.

If you wish to run the hooks against the entire repository, run:

```bash
pre-commit run --all-files
```

The other checks that are performed can be seen in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml). At the time of writing, these are:

- No direct committing to `main`, or `release/*`
- Check json for validity
- Ensure newline character at the end of files
- Trim end of line whitespace characters
- Check for no merge conflict lines accidentally being staged
- Clang format
- Python block (line length 100)


## Testing and Analysis

Several targets and CMake configuration options are defined as shown below to test the qe-compiler. The standard test suite consists of a format check to ensure the source follows the configured style defined in `.clang-format`, a run of `clang-tidy` to perform code analysis, and a run of the test suite. It is also possible to run additional analysis with Valgrind and the Google Sanitizers below. CMake commands should always be run from a build directory.

Our test suite contains both unit tests (using googletest) and integration tests that exercise the full qss-compiler or qss-opt tools (using LLVM lit).

### Debugging LIT Tests

A large fraction of the regression test suite is orchestrated using the LLVM Integrated Tester (LIT). When investigating test execution, several command line arguments for LIT may be helpful:

- `-vv` or `--echo-all-commands` to print all commands to stdout while execution, which implies `-v` / `--verbose` that prints test output on test failures
- `-a` or `--show-all` prints information about all tests during execution
- `--show-suites` and `--show-tests` to list all discovered test suites and tests; note that these two options exit immediately and will not run the tests.

There are several options for passing additional commands to lit:

- set the cmake variable `LIT_TEST_EXTRA_ARGS` or include them directly in `test/CMakeLists.txt` in the call to `add_lit_testsuite` so that they will be used for the targets `check-tests` and `check-all`
- run `ninja -v` and capture the command line  for `lit`, then modify and manually run that command line.

LIT tests may be filtered in order to limit the number of tests that are run. LIT supports the
`LIT_FILTER` and `LIT_FILTER_OUT` environment variables to filter in or out tests using a regex.
For example using `export LIT_FILTER=Conversion/CodeGen/pulse-gen-drive-0.mlir` will limit the lit
test to that single test. Note to use this with the `check-tests` target requires passing
`-DLIT_TEST_EXTRA_ARGS=--allow-empty-runs` to cmake otherwise the testing will stop on the first
target that has an empty run.

### Setting Paths for Manual Test Runs

The compiler may require static resources at runtime (e.g., target-specific tools or libraries). It will search for the resources in the path defined by the environment variable `QSSC_RESOURCES` or in the designated install location if that variable is not present.

For running `qss-compiler` or `qss-opt` from the build directory, you may use a generated shell script that takes care of setting up the required environment variables as follows:

```sh
# assuming current directory is the build directory
source qe-compiler/qssc-activate
# now call qss-compiler tool as usual
```

You may clean up your environment by calling `qssc-deactivate` (a shell function defined by the script `qssc-activate`).

The automated test runs take care of setting up the environment as required.

### Adding Unit Tests

Unit tests live in `test/unittest` and use the [GoogleTest framework](https://google.github.io/googletest/) for discovering and running test cases. When adding a new test, first consider whether it relates to an existing test and add it to an existing source file with test cases if that is the case. Otherwise, add your new `.cpp` source file in `test/unittest/CMakeLists.txt` to the definition of `unittest-qss-compiler` -- there should be a comment that points to the right place. Take a look at `test/unittest/test1.cpp` for a basic example.

The [Googletest Primer](https://google.github.io/googletest/primer.html) is a good place to get started. Also, take a look at our existing tests to get inspired. To learn more, have a look at the [comprehensive documentation of GoogleTest](https://google.github.io/googletest/).


## CI and Release Cycle
Please keep the following points in mind when developing:

- CI builds the repo on all branches all the time.
- CI builds using `conan` only.

### Branches

* `main`:
The main branch is used for the development of the next release.
It is updated frequently and should *not* be considered stable. On the development
branch, breaking changes can and will be introduced.
All efforts should be made to ensure that the development branch is maintained in
a self-consistent state that is passing continuous integration (CI).
Changes should not be merged unless they are verified by CI.
* `release/<major>.<minor>` branches:
Branches under `release/<major>.<minor>` are used to maintain released versions of the qe-compiler.
They contain the version of the compiler corresponding to the
release as identified by its [semantic version](https://semver.org/). For example,
`release/1.5` would be the compiler version for major version 1 and minor version 5.
On these branches, the compiler
is considered stable. The only changes that may be merged to a release branch are
patches/bugfixes. When a patch is required when possible the fix should
first be made to the development branch through a pull request.
The fix should then be backported from the development branch to the
target release branch (of name `release/<major>.<minor>`) by creating a pull request on
Github into the target release branch with the relevant cherry-picked commits.
The new release branch `HEAD` should be tagged (see [Tags](#tags)) with a new
`v<major.minor.patch>` version and pushed to Github.
* `<org>/release/<major>.<minor>` branches:
Some organizations (such as IBM) may require the creation of special release branches
that they will manage for various purposes. These should follow the same rules as
standard `release/` branches above but should be generally discouraged in favour of using
a fork to maintain separate release branches.

### Tags

*Note*: Only project administrators may perform these steps.

Git tags are used to tag the specific commit associated with a versioned release.
Tags must take the form of `v<major>.<minor>.<patch>-<labels>`. For example the semver
`v1.5.1` would point to the compiler release with major version 1,
minor version 5, and, patch version 1. The current development version would therefore be
`v1.6.0`. All official releases when tagged must always point to the current HEAD
of a release branch.

Tags for organizations should be of the form `<org>-v<major>.<minor>.<patch>`.


### Release cycle

To release a version a new version:

- (Option A) If releasing a major/minor version create a new release branch for the version (See [Branches](#branches)).
   This should be cut from the latest development branch.
   ```bash
   git checkout -b release/<version> <base>
   git push -u origin release/<version>
   ```
- (Option B) If releasing a patch version:
  -  checkout the existing release branch for your target major/minor version to apply the patch
   ```bash
   git checkout -b <backport>-<desc>-release/<version> release/<version>
   ```
  - Apply your changes (or cherry-pick existing commits) to your new branch and then push your branch to your Github fork
   ```bash
   git push -u origin <your-branch>
   ```
  - Make a PR from your new branch on your fork into the target `release/<version>` branch with the form `[Backport] <Title>` and merge the PR
- Create a new tag with the required semantic version number (see [Tags](#tags)), tagging the `HEAD` of the target `release/<version>` branch.
  Push the tag to Github which will trigger CI.
    ```bash
    git tag -a v<version> -m "<description> e.g. release v<x>.<y>.<z>" # <- where version is the version number of the tag.
    git push -u origin v<version>
    ```

### Example release cycle

For this example assume the current release of the qe-compiler is version `v0.5.1`. This will correspond to a commit
on `release/0.5`. The project's development branch reflects the development state of the next release - `v0.6.0`.

To trigger a bugfix release - `v0.5.2`:
1. Create a PR into `release/0.5` with all required changes. The PR ideally should begin with title of the form `[Backport] <Title>`.
   These may be backported commits from `main`.
2. Upon merger of the PR tag the HEAD of `release/0.5` with `vv0.5.2` and push to Github.

To trigger a minor release - `v0.6.0`:
1. Create a new release branch `release/0.6` using the current development branch (`main`) as the base branch, eg., `git checkout -b release/0.6 main`.
2. Push this branch to Github.
3. Tag the branch with `v0.6.0` and push to Github.

## Notes

### Conan dependencies

Most of our dependencies live in our [conan-center](https://conan.io/center).

#### Bumping dependency versions:

- `qasm`: Add a new version [here](https://github.com/openqasm/qe-compiler/blob/main/conandata.yml#L9). You will also need to updated the vendored [conan package](https://github.com/openqasm/qe-compiler/tree/main/conan/qasm).

- `llvm`: Trigger a new push [here](https://github.com/openqasm/qe-compiler/blob/main/conandata.yml#L8). You will also need to updated the vendored [conan package](https://github.com/openqasm/qe-compiler/tree/main/conan/llvm).

- `clang-tools-extra`: Trigger a new push [here](https://github.com/openqasm/qe-compiler/blob/main/conandata.yml#L7). You will also need to updated the vendored [conan package](https://github.com/openqasm/qe-compiler/tree/main/conan/clang-tools-extra).
