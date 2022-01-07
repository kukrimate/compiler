fn trap(r1: u16, r2: u16, r3: u16);

fn putchar(ch: u16) {
	trap(2, &ch as u16, 1);
}


fn puts(s: *u16) {
	// Gross hack, pointer to a stupidly long array
	// K doesn't have slices yet, and C style strings
	// really are a hack in their current form
	let aptr = s as *[u16; 10000];

	let i = 0;
	while ((*aptr)[i] != 0) {
		putchar((*aptr)[i]);
		i = i + 1;
	}
}


fn main() {
	puts("Hello World!");
}
