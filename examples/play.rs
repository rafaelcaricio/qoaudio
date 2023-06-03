use qoaudio::{QoaDecoder, QoaRodioSource};
use rodio::{OutputStream, Sink, Source};
use std::thread::sleep;

fn main() {
    let Some(audio_path) = std::env::args().nth(1) else {
        eprintln!("Usage: play <path to qoa file>");
        return;
    };

    let decoder = QoaDecoder::open(audio_path).expect("open file");
    let audio = QoaRodioSource::new(decoder);

    println!("Decoded header:");
    println!("\tchannels: {}", audio.channels());
    println!("\tsample rate: {}", audio.sample_rate());
    println!("\tDuration: {:?}", audio.total_duration());

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    println!("Playing...");
    let dur = audio.total_duration().unwrap();
    sink.append(audio);
    sleep(dur);
}
