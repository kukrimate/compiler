record $a {
    $a: u8,
    $b: u32,
};

union $b {
    $a: i32,
    $c: u8,
};

static $myarr: *u8 = u8 { 5,5,5,5,5,5 };
static $myarr2: *u8 = u8[10];

static $global_5: u8 = 5;
static $mystring: *u8 = u8"mystring";
static export $mychar: u8 = u8'a':

fn export $main($argc: i32, $argv: **u8) {
    auto $ui: u8;

    auto $bar: i32 = 5i32;
    auto $foo: i32 = $bar;

    jeq @a, $bar, 5i32;
    jl @a, $bar, 10i32;
    jg @a, $bar, 15i32;

@a:
    static $foobar: i16 = 5i16;
    jmp @a;

    set $bar = 6i32;
    set $foo = 12i32;
}
