#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(decoder) = qoaudio::QoaDecoder::new(std::io::Cursor::new(data)) else {
        return
    };
    decoder.take_while(|i| matches!(i, Ok(_))).count();
});
