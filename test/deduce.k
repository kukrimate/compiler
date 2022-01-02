fn extern printf(str: *u8, ...) -> i32;

fn export main() {
    auto a = 5i32;
    auto b;
    auto c;
    auto d = 1;

    set c = d;
    set b = c;
    set a = b;

    eval printf("a = %d\n", a);
}

fn f() {
    auto arr1: [i32; 2] = [ 1, 2 ];

    auto arr2 = [1,2,3];
    set arr2[1] = 5i32;
}
