// gtest-lite: a minimal, dependency-free subset of GoogleTest for the
// nexus-am bare-metal target.
//
// Real GoogleTest cannot link against klib (its libstdc++ pulls ~190
// version-tagged glibc symbols). This header implements only the macros the
// test suite uses, with printf-based reporting that mimics gtest output so
// ctest can match on "[ PASSED" / "[ FAILED" in the summary lines.
//
// Supported: TEST, TEST_F, TEST_P, ASSERT_EQ, EXPECT_EQ, ASSERT_TRUE,
// EXPECT_TRUE, TestWithParam<T>, Values(...), Combine(...),
// INSTANTIATE_TEST_SUITE_P, RUN_ALL_TESTS, ::testing::InitGoogleTest, and
// streaming a trailing message via `ASSERT_EQ(a,b) << "text"`.
//
// Parameterized tests: TEST_P defines a subclass whose TestBody() reads
// GetParam(). INSTANTIATE_TEST_SUITE_P expands the Values/Combine generator
// at static-init time and registers one registry entry per combination; each
// entry's thunk stores its parameter into the fixture's static slot and then
// runs the body.
//
// There is no exception support on this target, so a failed ASSERT aborts
// the current test via an early return (same observable behavior as gtest's
// fatal assertion for these tests).

#pragma once

#include "riscv_port.hpp"

#include <cstddef>
#include <tuple>
#include <utility>

namespace testing
{

// ---- test registry -------------------------------------------------------

using TestFunc = void (*)();

struct TestInfo
{
    const char* suite;
    const char* name;
    TestFunc run;
    TestInfo* next;
};

inline TestInfo*& RegistryHead()
{
    static TestInfo* head = nullptr;
    return head;
}

inline bool Register(TestInfo* info)
{
    info->next = RegistryHead();
    RegistryHead() = info;
    return true;
}

// ---- failure bookkeeping -------------------------------------------------

inline int& CurrentTestFailures()
{
    static int failures = 0;
    return failures;
}

inline void ReportFailure(const char* file, int line, const char* expr,
                          long long expected, long long actual)
{
    ++CurrentTestFailures();
    printf("%s:%d: Failure\n  Expected: %s\n    Which is: %lld\n"
           "  Actual  : %lld\n",
           file, line, expr, expected, actual);
}

// A sink that swallows `<< "message"` after an assertion macro. Each << call
// just prints the fragment; returning a reference allows chaining.
struct MessageSink
{
    template <typename T>
    MessageSink& operator<<(const T&)
    {
        return *this;
    }
};

// ---- runner --------------------------------------------------------------

// Records which TestInfo is currently running so a parameterized thunk can
// find its own entry (function pointers cannot capture).
inline TestInfo*& ActiveTest()
{
    static TestInfo* active = nullptr;
    return active;
}

inline int RunAllTests()
{
    // Registry is a stack (LIFO); reverse so tests run in declaration order.
    TestInfo* reversed = nullptr;
    for (TestInfo* t = RegistryHead(); t; )
    {
        TestInfo* next = t->next;
        t->next = reversed;
        reversed = t;
        t = next;
    }

    int total = 0;
    int failed = 0;
    for (TestInfo* t = reversed; t; t = t->next)
    {
        ++total;
        CurrentTestFailures() = 0;
        ActiveTest() = t;
        printf("[ %-24s ] %s.%s\n", "RUN", t->suite, t->name);
        t->run();
        if (CurrentTestFailures() == 0)
        {
            printf("[ %-24s ] %s.%s\n", "OK", t->suite, t->name);
        }
        else
        {
            ++failed;
            printf("[ %-24s ] %s.%s (%d failure%s)\n", "FAILED", t->suite, t->name,
                   CurrentTestFailures(), CurrentTestFailures() == 1 ? "" : "s");
        }
    }

    printf("[ %-24s ] %d test%s ran.\n", "==========", total, total == 1 ? "" : "s");
    if (failed == 0)
    {
        printf("[ %-24s ] %d test%s.\n", "PASSED", total, total == 1 ? "" : "s");
    }
    else
    {
        printf("[ %-24s ] %d test%s failed.\n", "FAILED", failed, failed == 1 ? "" : "s");
    }
    return failed == 0 ? 0 : 1;
}

inline void InitGoogleTest(int*, char**) {}

// ---- base classes --------------------------------------------------------

// Both plain fixtures and parameterized fixtures derive from this so the
// generated subclass can override TestBody().
class Test
{
public:
    virtual ~Test() = default;
    virtual void TestBody() = 0;
};

template <typename T>
class TestWithParam : public Test
{
public:
    using ParamType = T;

    static const T& GetParam() { return *current_; }
    static void SetParam(const T& p) { current_ = &p; }

private:
    static const T* current_;
};

template <typename T>
const T* TestWithParam<T>::current_ = nullptr;

// ---- parameter generators ------------------------------------------------

template <typename T, typename... Ts>
struct ValuesHolder
{
    using TupleType = std::tuple<T, Ts...>;
    static constexpr std::size_t kCount = 1 + sizeof...(Ts);
    TupleType values;
};

template <typename T, typename... Ts>
constexpr ValuesHolder<T, Ts...> Values(T first, Ts... rest)
{
    return {std::make_tuple(first, rest...)};
}

template <typename... Holders>
struct CombineHolder
{
    std::tuple<Holders...> holders;
};

template <typename... Holders>
constexpr CombineHolder<Holders...> Combine(Holders... holders)
{
    return {std::make_tuple(holders...)};
}

} // namespace testing

// ---- instantiation machinery (internal) ----------------------------------

namespace gtest_lite_detail
{

// Name buffers: "prefix/suite.index", one per registered test. Shared by every fixture of the
// binary, so the aggregate unit_tests image needs room for all of them (K1..K6 + Add: ~350).
// Past the end, all names share one spare buffer instead of overrunning the array.
inline char* NameBuffer(unsigned slot)
{
    static char names[1024][96];
    static char spare[96];
    return slot < 1024 ? names[slot] : spare;
}

inline unsigned& NameSlots()
{
    static unsigned n = 0;
    return n;
}

inline const char* MakeName(const char* prefix, const char* suite,
                            unsigned index)
{
    char* buf = NameBuffer(NameSlots()++);
    char* p = buf;
    for (const char* s = prefix; *s; ++s) *p++ = *s;
    *p++ = '/';
    for (const char* s = suite; *s; ++s) *p++ = *s;
    *p++ = '.';
    if (index >= 100) *p++ = static_cast<char>('0' + (index / 100) % 10);
    if (index >= 10)  *p++ = static_cast<char>('0' + (index / 10) % 10);
    *p++ = static_cast<char>('0' + index % 10);
    *p = '\0';
    return buf;
}

// "prefix/suite.body/index" — gtest's parameterized test naming.
inline const char* MakeFullName(const char* prefix, const char* suite,
                                const char* body, unsigned index)
{
    char* buf = NameBuffer(NameSlots()++);
    char* p = buf;
    for (const char* s = prefix; *s; ++s) *p++ = *s;
    *p++ = '/';
    for (const char* s = suite; *s; ++s) *p++ = *s;
    *p++ = '.';
    for (const char* s = body; *s; ++s) *p++ = *s;
    *p++ = '/';
    if (index >= 100) *p++ = static_cast<char>('0' + (index / 100) % 10);
    if (index >= 10)  *p++ = static_cast<char>('0' + (index / 10) % 10);
    *p++ = static_cast<char>('0' + index % 10);
    *p = '\0';
    return buf;
}

// Per-fixture registration state.
//
// Each TEST_P body for a fixture registers a "body runner" (a function that
// news up the generated subclass and runs TestBody) into BodyList. Then
// INSTANTIATE_TEST_SUITE_P expands the parameter generator and, for each
// (body, param) pair, registers one TestInfo whose thunk loads the param into
// the fixture's static slot and invokes the body runner.
template <typename Fixture>
struct ParamState
{
    using ParamType = typename Fixture::ParamType;
    using BodyFunc = void (*)();

    // Registered TEST_P bodies for this fixture.
    static BodyFunc bodies[64];
    static const char* body_names[64];
    static unsigned body_count;

    // One entry per (body, param) combination.
    static ParamType params[128];
    static BodyFunc entry_body[128];
    static ::testing::TestInfo infos[128];
    static unsigned used;

    static void Thunk()
    {
        const ::testing::TestInfo* active = ::testing::ActiveTest();
        for (unsigned i = 0; i < used; ++i)
        {
            if (&infos[i] == active)
            {
                Fixture::SetParam(params[i]);
                entry_body[i]();
                return;
            }
        }
    }

    // Called by TEST_P to register a body. Returns true for static-init.
    static bool RegisterBody(const char* name, BodyFunc fn)
    {
        if (body_count >= 64)
        {
            printf("gtest_lite: too many TEST_P bodies\n");
            return false;
        }
        bodies[body_count] = fn;
        body_names[body_count] = name;
        ++body_count;
        return true;
    }

    // Called by INSTANTIATE_TEST_SUITE_P for each parameter combination.
    static void Register(const char* prefix, const char* suite,
                         unsigned index, const ParamType& value)
    {
        for (unsigned b = 0; b < body_count; ++b)
        {
            if (used >= 128)
            {
                printf("gtest_lite: too many instantiations\n");
                return;
            }
            params[used] = value;
            entry_body[used] = bodies[b];
            infos[used].suite = suite;
            // Name: "prefix/suite.body/index" (gtest-style).
            infos[used].name = MakeFullName(prefix, suite, body_names[b], index);
            infos[used].run = &Thunk;
            infos[used].next = nullptr;
            ::testing::Register(&infos[used]);
            ++used;
        }
    }
};

template <typename Fixture>
typename ParamState<Fixture>::BodyFunc ParamState<Fixture>::bodies[64];

template <typename Fixture>
const char* ParamState<Fixture>::body_names[64];

template <typename Fixture>
unsigned ParamState<Fixture>::body_count = 0;

template <typename Fixture>
typename ParamState<Fixture>::ParamType ParamState<Fixture>::params[128];

template <typename Fixture>
typename ParamState<Fixture>::BodyFunc ParamState<Fixture>::entry_body[128];

template <typename Fixture>
::testing::TestInfo ParamState<Fixture>::infos[128];

template <typename Fixture>
unsigned ParamState<Fixture>::used = 0;

// Register every element of a Values list (single-axis parameterization).
// Uses a C++17 fold over the comma operator — std::initializer_list's backing
// array is unreliable under this freestanding link, so we avoid it.
template <typename Fixture, typename Tuple, std::size_t... I>
void RegisterValues(const char* prefix, const char* suite, const Tuple& values,
                    std::index_sequence<I...>, unsigned& index)
{
    (ParamState<Fixture>::Register(prefix, suite, index++, std::get<I>(values)), ...);
}

// Cartesian product of a Combine's Values lists. Each product element is a
// std::tuple, which becomes the fixture's ParamType.
//
// Implementation: nested recursion over the holder pack. CombineRecurse()
// peels one holder off the pack and calls CombineAxis() to iterate its
// values; when the pack is empty the accumulated tuple is registered.
template <typename Fixture, typename Acc, typename HeadHolder,
          typename... RestHolders, std::size_t... I>
void CombineAxis(const char* prefix, const char* suite, unsigned& index,
                 const Acc& acc, const HeadHolder& holder,
                 const std::index_sequence<I...>, const RestHolders&... rest);

template <typename Fixture, typename Acc>
void CombineRecurse(const char* prefix, const char* suite, unsigned& index,
                    const Acc& acc)
{
    ParamState<Fixture>::Register(prefix, suite, index++, acc);
}

template <typename Fixture, typename Acc, typename HeadHolder,
          typename... RestHolders>
void CombineRecurse(const char* prefix, const char* suite, unsigned& index,
                    const Acc& acc, const HeadHolder& holder,
                    const RestHolders&... rest)
{
    CombineAxis<Fixture>(prefix, suite, index, acc, holder,
                         std::make_index_sequence<HeadHolder::kCount>{},
                         rest...);
}

template <typename Fixture, typename Acc, typename HeadHolder,
          typename... RestHolders, std::size_t... I>
void CombineAxis(const char* prefix, const char* suite, unsigned& index,
                 const Acc& acc, const HeadHolder& holder,
                 const std::index_sequence<I...>, const RestHolders&... rest)
{
    (CombineRecurse<Fixture>(prefix, suite, index,
         std::tuple_cat(acc, std::make_tuple(std::get<I>(holder.values))),
         rest...), ...);
}

// Entry point: dispatch on whether the generator is a Values or a Combine.
template <typename Fixture, typename T, typename... Ts>
void Instantiate(const char* prefix, const char* suite,
                 const ::testing::ValuesHolder<T, Ts...>& holder)
{
    unsigned index = 0;
    RegisterValues<Fixture>(prefix, suite, holder.values,
        std::make_index_sequence<1 + sizeof...(Ts)>{}, index);
}

template <typename Fixture, typename... Holders>
void Instantiate(const char* prefix, const char* suite,
                 const ::testing::CombineHolder<Holders...>& holder)
{
    unsigned index = 0;
    std::apply(
        [&](const Holders&... h) {
            CombineRecurse<Fixture>(prefix, suite, index, std::make_tuple(), h...);
        },
        holder.holders);
}

} // namespace gtest_lite_detail

// ---- macros --------------------------------------------------------------

#define GTEST_LITE_CONCAT_INNER(a, b) a##b
#define GTEST_LITE_CONCAT(a, b) GTEST_LITE_CONCAT_INNER(a, b)

// TEST(suite, name)
#define TEST(test_suite_name, test_name)                                        \
    static void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_suite_name, _),        \
        GTEST_LITE_CONCAT(test_name, _body))();                                 \
    static ::testing::TestInfo GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(             \
        test_suite_name, _), GTEST_LITE_CONCAT(test_name, _info)) = {           \
        #test_suite_name, #test_name,                                           \
        &GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_suite_name, _),               \
            GTEST_LITE_CONCAT(test_name, _body)),                               \
        nullptr};                                                               \
    static bool GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_suite_name, _),        \
        GTEST_LITE_CONCAT(test_name, _registered)) =                            \
        ::testing::Register(&GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(               \
            test_suite_name, _), GTEST_LITE_CONCAT(test_name, _info)));         \
    static void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_suite_name, _),        \
        GTEST_LITE_CONCAT(test_name, _body))()

// TEST_F(fixture, name): define a subclass with the body and register it
// directly (no parameter).
#define TEST_F(test_fixture, test_name)                                         \
    class GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                 \
        GTEST_LITE_CONCAT(test_name, _test)) : public test_fixture              \
    {                                                                           \
    public:                                                                     \
        void TestBody() override;                                               \
    };                                                                          \
    static void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),           \
        GTEST_LITE_CONCAT(test_name, _run))()                                   \
    {                                                                           \
        GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                   \
            GTEST_LITE_CONCAT(test_name, _test)) instance;                      \
        instance.TestBody();                                                    \
    }                                                                           \
    static ::testing::TestInfo GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(             \
        test_fixture, _), GTEST_LITE_CONCAT(test_name, _info)) = {              \
        #test_fixture, #test_name,                                              \
        &GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                  \
            GTEST_LITE_CONCAT(test_name, _run)),                                \
        nullptr};                                                               \
    static bool GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),           \
        GTEST_LITE_CONCAT(test_name, _registered)) =                            \
        ::testing::Register(&GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(               \
            test_fixture, _), GTEST_LITE_CONCAT(test_name, _info)));            \
    void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                  \
        GTEST_LITE_CONCAT(test_name, _test))::TestBody()

// TEST_P(fixture, name): define the subclass and register a body-runner with
// the fixture's ParamState. INSTANTIATE_TEST_SUITE_P later pairs each body
// with every parameter combination.
#define TEST_P(test_fixture, test_name)                                         \
    class GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                 \
        GTEST_LITE_CONCAT(test_name, _test)) : public test_fixture              \
    {                                                                           \
    public:                                                                     \
        void TestBody() override;                                               \
    };                                                                          \
    static void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),           \
        GTEST_LITE_CONCAT(test_name, _run))()                                   \
    {                                                                           \
        GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                   \
            GTEST_LITE_CONCAT(test_name, _test)) instance;                      \
        instance.TestBody();                                                    \
    }                                                                           \
    static bool GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),           \
        GTEST_LITE_CONCAT(test_name, _body_registered)) =                       \
        ::gtest_lite_detail::ParamState<test_fixture>::RegisterBody(            \
            #test_name,                                                         \
            &GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),              \
                GTEST_LITE_CONCAT(test_name, _run)));                           \
    void GTEST_LITE_CONCAT(GTEST_LITE_CONCAT(test_fixture, _),                  \
        GTEST_LITE_CONCAT(test_name, _test))::TestBody()

// INSTANTIATE_TEST_SUITE_P(prefix, suite, generator)
#define INSTANTIATE_TEST_SUITE_P(prefix, test_suite_name, generator)            \
    static bool GTEST_LITE_CONCAT(prefix, GTEST_LITE_CONCAT(_,                  \
        GTEST_LITE_CONCAT(test_suite_name, _instantiated))) = [] {              \
        ::gtest_lite_detail::Instantiate<test_suite_name>(                      \
            #prefix, #test_suite_name, generator);                              \
        return true;                                                            \
    }();

#define RUN_ALL_TESTS() ::testing::RunAllTests()

// ---- assertions ----------------------------------------------------------

#define GTEST_LITE_ASSERT_EQ(expected, actual)                                  \
    do                                                                          \
    {                                                                           \
        const auto gtest_lite_e = (expected);                                   \
        const auto gtest_lite_a = (actual);                                     \
        if (!(gtest_lite_e == gtest_lite_a))                                    \
        {                                                                       \
            ::testing::ReportFailure(__FILE__, __LINE__, #expected " == "       \
                #actual, static_cast<long long>(gtest_lite_e),                  \
                static_cast<long long>(gtest_lite_a));                          \
            return;                                                             \
        }                                                                       \
    } while (0);                                                                \
    ::testing::MessageSink()

#define ASSERT_EQ(expected, actual) GTEST_LITE_ASSERT_EQ(expected, actual)

#define EXPECT_EQ(expected, actual)                                             \
    do                                                                          \
    {                                                                           \
        const auto gtest_lite_e = (expected);                                   \
        const auto gtest_lite_a = (actual);                                     \
        if (!(gtest_lite_e == gtest_lite_a))                                    \
        {                                                                       \
            ::testing::ReportFailure(__FILE__, __LINE__, #expected " == "       \
                #actual, static_cast<long long>(gtest_lite_e),                  \
                static_cast<long long>(gtest_lite_a));                          \
        }                                                                       \
    } while (0);                                                                \
    ::testing::MessageSink()

#define ASSERT_TRUE(cond)                                                       \
    do                                                                          \
    {                                                                           \
        if (!(cond))                                                            \
        {                                                                       \
            ::testing::ReportFailure(__FILE__, __LINE__, #cond, 1, 0);          \
            return;                                                             \
        }                                                                       \
    } while (0);                                                                \
    ::testing::MessageSink()

#define EXPECT_TRUE(cond)                                                       \
    do                                                                          \
    {                                                                           \
        if (!(cond))                                                            \
        {                                                                       \
            ::testing::ReportFailure(__FILE__, __LINE__, #cond, 1, 0);          \
        }                                                                       \
    } while (0);                                                                \
    ::testing::MessageSink()
