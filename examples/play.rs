use qoaudio::{DecodedAudio, QoaDecoder};
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
    let mut file = File::open(audio_path).unwrap();
    file.read_to_end(&mut bytes).unwrap();

    let mut qoa = QoaDecoder::decode_header(&bytes).unwrap();
    // let mut qoa = QoaDecoder::streaming();
    let audio: DecodedAudio = qoa.decode_frames(&bytes).unwrap().try_into().unwrap();
    println!("Decoded header:");
    println!("\tchannels: {}", audio.channels());
    println!("\tsample rate: {}", audio.sample_rate());
    println!("\tDuration: {:?}", audio.duration());

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    println!("Playing...");
    let dur = audio.duration();
    sink.append(audio);
    sleep(dur);
}
