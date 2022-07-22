//! midi端口控制模块

use midir::{ConnectError, MidiOutput, MidiOutputConnection};

extern crate midir;

/// 发声信息
const NOTE_ON_MSG: u8 = 0x90;
/// 停止发声信息
const NOTE_OFF_MSG: u8 = 0x80;
/// 音符发声强度
const VELOCITY: u8 = 0x64;

/// midi端口连接
pub struct MidiConn(MidiOutputConnection);

impl MidiConn {
    /// 连接midi端口
    pub fn new(id: usize) -> Result<Self, ConnectError<MidiOutput>> {
        // 获得midi输出端口
        let out = MidiOutput::new("midiOutPut").unwrap();
        let port = &out.ports()[id];
        // 返回端口连接的实例
        out.connect(port, &format!("midi port {:}", id))
            .map(|conn| Self(conn))
    }

    /// 发送音符发声
    pub fn play_on(&mut self, pitch: u8) -> Result<(), midir::SendError> {
        self.0.send(&[NOTE_ON_MSG, pitch, VELOCITY])
    }

    /// 停止音符发声
    pub fn play_off(&mut self, pitch: u8) -> Result<(), midir::SendError> {
        self.0.send(&[NOTE_OFF_MSG, pitch, VELOCITY])
    }
}
