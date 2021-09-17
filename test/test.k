fn extern $gets($s: *u8) -> *u8;

fn export $main() {
    auto $arr: [u8; 100u32];
    eval $gets(&$arr);

    auto $i: u8 = 10u8 * 50u8;
}
