static $myarr: [u8;6u8] = { 5u8, 5u8, 5u8, 5u8, 5u8, 5u8 };
static $myarr2: [u8;5i32];

// static $mystring: *u8 = "mystring" u8;

record $a {
    $a: u8,
    $b: u32,
};

union $b {
    $a: i32,
    $c: u8,
};

static $global_5: u8 = 5 u8;
// static $mystring: *u8 = "mystring"u8;
static export $mychar: u8 = 'a'u8;

fn export $main($argc: i32, $argv: **u8) {
    auto $ui: u8; // uninitailized variable
    set $ui = 5u8;

    auto $bar: i32 = 5i32; // initialized auto storage
    auto $foo: i32 = (5i32 + $bar) & $ui as i32;

    jeq @a, $bar, 5i32; // jumps based on comparisons
    jl @a, $bar, 10i32;
    jg @a, $bar, 15i32;

@a:
    jmp @a; // unconditional jump

    set $bar = 6i32; // re-assingment (if I ever do SSA there will be phis instead)
    set $foo = 12i32;
}
