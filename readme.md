# compiler-rs
Compiler for a simple programming language, written in Rust.

The grammar of the language is defined in (something resembling) EBNF
notation in `doc/grammar.txt`.

Current features:
- Strong typing, no implicit type conversions
- Signed and unsigned integer types of various widths
- Record (similar to C struct) types
- C like expressions
- Block based control flow
- Rust inspired sytax
- Assembly generation for x86\_64 (SysV ABI)

Notably missing features:
- Any "real" optimizations (we do constant folding)
- I would currently categorize the generated assembly simply as "bad"
- Pointer arithmetic (can be circumvented using casts)
- Struct return types and arguments (struct pointers work fine)
- Floating point math
- Many other bugs and TODOs

## license
GPLv2 only.

## so you want to also write a compiler? (aka reading list and inspirations)
- Wikipedia articles on:
  * Regular langauges and lexical analysis
  * Context free languages, more specifically LL, LR grammars and shift reduce
    and recursive descent parsers
  * This compiler uses an auto generated lexical analyzer and predicitive
    recursive-descent parser for its LL(1) grammar
- Compilers: Principles, Techniques, and Tools (aka the Dragon Book)
  * Notably my boolean code generator is roughly based on an algorithm from here
- Neat C11 compiler: https://github.com/rui314/chibicc
  * Educational compiler for C11 written in C itself
  * Also builds an AST, uses a similarly awful code generator
- Less neat, but single pass compiler for C99: https://bellard.org/tcc/
  * Aka how to write a compiler if you are more clever than me
  * No AST is built, code is generated directly by semantic actions embedded in
    its (more-or-less, C isn't a context free language) recursive-decent parser
  * Better code generator with registerized temporaries
- This project: on how to not do things
  * Currently there is no purpose for the AST, it's just traversed the same way
    it was built, this compiler could easily be single pass
  * A modern compiler needs at least register temporaries and basic optimization,
    the generated code is absolutely awful
