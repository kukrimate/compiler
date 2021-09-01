# compiler-rs
Compiler backend written in Rust. It compiles an IL to x86_64 machine code.

The IL uses a high-level expression syntax, but no block based control flow.
It's strongly typed with no implicit type conversions, and a Rust inspired syntax.

Current features:
- Non-SSA IL with most features needed to compile C-like languages
- Code generator for x86_64 (SysV ABI)

Missing features:
- Register allocation (accumulator + stack evaluation)
- Any sort of optimization (other than constant folding)
- Struct return types and arguments (struct pointers work fine)
- Variable argument support
- Floating point math

# license
GPLv2 only.
