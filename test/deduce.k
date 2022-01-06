fn extern printf(str: *u8, ...) -> i32;

fn export main() {
    let a;
    let b = 5i32;
    let c;
    let d = 1;

    c = d;
    b = c;
    a = b;

    printf("a = %d\n", a);
}

fn f() {
    let arr1: [i32; 2] = [ 1, 2 ];

    let arr2 = [1,2,3];
    arr2[1] = 5i32;
}

fn g() {
    let a;
    let b;

    a = a + b;
}
