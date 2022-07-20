use midir::{ConnectError, MidiOutput, MidiOutputConnection};

extern crate midir;

const NOTE_ON_MSG: u8 = 0x90;
const NOTE_OFF_MSG: u8 = 0x80;
const VELOCITY: u8 = 0x64;

pub struct MidiConn(MidiOutputConnection);

impl MidiConn {
    pub fn new(id: usize) -> Result<Self, ConnectError<MidiOutput>> {
        let out = MidiOutput::new("midiOutPut").unwrap();
        let port = &out.ports()[id];
        out.connect(port, &format!("midi port {:}", id))
            .map(|conn| Self(conn))
    }

    pub fn play_on(&mut self, pitch: u8) -> Result<(), midir::SendError> {
        self.0.send(&[NOTE_ON_MSG, pitch, VELOCITY])
    }

    pub fn play_off(&mut self, pitch: u8) -> Result<(), midir::SendError> {
        self.0.send(&[NOTE_OFF_MSG, pitch, VELOCITY])
    }
}
