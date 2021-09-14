fn $foo() {
    set *(0xdeadbeef u64 as *u8) = 55 u8;
}
