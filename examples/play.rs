use qoaudio::QoaDecoder;
use rodio::{OutputStream, Sink};
use std::fs::File;
use std::io::Read;
use std::thread::sleep;

fn main() {
    let Some(audio_path) = std::env::args().nth(1) else {
        eprintln!("Usage: play <path to qoa file>");
        return;
    };

    let mut bytes = Vec::new();
    let mut file = File::open(&audio_path).unwrap();
    file.read_to_end(&mut bytes).unwrap();

    let qoa = QoaDecoder::new(&bytes).unwrap();
    println!("Decoded header:");
    println!("\tchannels: {}", qoa.channels());
    println!("\tsamplerate: {}", qoa.sample_rate());
    println!("\tsamples: {}", qoa.samples());
    println!("\tDuration: {:?}", qoa.duration());

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    println!("Playing...");
    let dur = qoa.duration();
    sink.append(qoa);
    sleep(dur);
}
