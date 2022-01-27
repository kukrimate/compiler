// SPDX-License-Identifier: GPL-2.0-only

//
// Compiler utilities
//

use std::mem::MaybeUninit;

/// Fixed length first-in first-out buffer
pub struct FIFO<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    pos: usize,
    len: usize,
}

impl<T, const N: usize> FIFO<T, N> {
    pub fn new() -> Self {
        FIFO {
            array: unsafe { MaybeUninit::uninit().assume_init() },
            pos: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        if i < self.len {
            Some(unsafe {
                &*self.array
                    // Get reference to the ith MaybeUninit cell
                    .get_unchecked((self.pos + i) % N)
                    // Get pointer to the inside of the cell
                    .as_ptr()
            })
        } else {
            None
        }
    }

    pub fn push(&mut self, val: T) {
        // Make sure there is room
        assert!(self.len < N);

        unsafe {
            self.array
                // Get reference to MaybeUninit cell
                .get_unchecked_mut((self.pos + self.len) % N)
                // Write value to cell
                .as_mut_ptr().write(val);
        }

        // Incrase length
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            let val = unsafe {
                self.array
                    // Get reference to MaybeUninit cell
                    .get_unchecked(self.pos)
                    // Read value from cell
                    .as_ptr().read()
            };

            // Increase position
            self.pos = (self.pos + 1) % N;
            // Decrease length
            self.len -= 1;

            Some(val)
        } else {
            // No elements to pop
            None
        }
    }
}

/// Peekable iterator adapter with fixed length peek window
pub struct PeekIter<T, const N: usize> where T: Iterator {
    iter: T,
    fifo: FIFO<T::Item, N>
}

impl<T, const N: usize> PeekIter<T, N> where T: Iterator {
    pub fn new(iter: T) -> Self {
        PeekIter {
            iter: iter,
            fifo: FIFO::new(),
        }
    }

    pub fn peek(&mut self, i: usize) -> Option<&T::Item> {
        while self.fifo.len() <= i {
            self.fifo.push(self.iter.next()?)
        }
        self.fifo.get(i)
    }
}

impl<T, const N: usize> Iterator for PeekIter<T, N> where T: Iterator {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.fifo.len() > 0 {
            self.fifo.pop()
        } else {
            self.iter.next()
        }
    }
}
