mod midi;

use bevy::{core::Stopwatch, prelude::*};
use midi::MidiConn;

const FALL_VECTOR: f64 = 200.0;

const NOTES_TABLE: [(KeyCode, u8, Color); 13] = [
    (KeyCode::A, 60, Color::WHITE),         // 1
    (KeyCode::W, 61, Color::BLACK),         // 1#
    (KeyCode::S, 62, Color::WHITE),         // 2
    (KeyCode::E, 63, Color::BLACK),         // 2#
    (KeyCode::D, 64, Color::WHITE),         // 3
    (KeyCode::F, 65, Color::WHITE),         // 4
    (KeyCode::R, 66, Color::BLACK),         // 4#
    (KeyCode::J, 67, Color::WHITE),         // 5
    (KeyCode::U, 68, Color::BLACK),         // 5#
    (KeyCode::K, 69, Color::WHITE),         // 6
    (KeyCode::I, 70, Color::BLACK),         // 6#
    (KeyCode::L, 71, Color::WHITE),         // 7
    (KeyCode::Semicolon, 72, Color::WHITE), // 8
];

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(WindowDescriptor {
            width: WIDTH,
            height: HEIGHT,
            title: String::from("Piano Player"),
            present_mode: bevy::window::PresentMode::Fifo,
            resizable: false,
            ..default()
        })
        .insert_resource(RecordNotes(vec![]))
        .insert_resource(RecordDuration::default())
        .add_state(GameStates::Record)
        .add_startup_system(setup)
        .add_system_set(SystemSet::on_enter(GameStates::Record).with_system(record_start))
        .add_system_set(
            SystemSet::on_update(GameStates::Record)
                .with_system(record_tick)
                .with_system(record_note),
        )
        .add_system(state_switch)
        .add_system(play_note)
        .run();
}

// Resources
#[derive(Debug)]
struct RecordNotes(Vec<Note>);

#[derive(Debug, Default)]
struct RecordDuration {
    time: Stopwatch,
}

// State
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum GameStates {
    Record,
    Playing,
}

// Components
#[derive(Component, Debug)]
struct Note {
    /// 音高
    pitch: u8,
    // velocity:
    /// 间隔
    step: f32,
    /// 时值
    duration: f32,
}

#[derive(Component, Default, Debug)]
struct PlayDuration {
    time: Stopwatch,
}

#[derive(Component, Debug)]
struct PianoKey {
    key: KeyCode,
    pitch: u8,
}

// Systems
fn setup(mut commands: Commands) {
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
    let key_width = WIDTH / NOTES_TABLE.len() as f32;
    for (idx, (key, pitch, color)) in NOTES_TABLE.iter().enumerate() {
        let x_position = key_width / 2.0 - WIDTH / 2.0 + idx as f32 * key_width;
        commands
            .spawn_bundle(SpriteBundle {
                sprite: Sprite {
                    color: *color,
                    custom_size: Some(Vec2::new(key_width - 2.0, 100.0)),
                    ..default()
                },
                transform: Transform::from_xyz(x_position, 50.0 - HEIGHT / 2.0, 0.0),
                ..default()
            })
            .insert(PianoKey {
                key: *key,
                pitch: *pitch,
            })
            .insert(PlayDuration::default());
    }
}

fn state_switch(mut keys: ResMut<Input<KeyCode>>, mut state: ResMut<State<GameStates>>) {
    if keys.just_pressed(KeyCode::Q) {
        match state.current() {
            GameStates::Record => state.set(GameStates::Playing).unwrap(),
            GameStates::Playing => state.set(GameStates::Record).unwrap(),
        }
        keys.reset(KeyCode::Q);
    }
}

fn play_note(keys: Res<Input<KeyCode>>, mut piano_keys: Query<(&mut Transform, &PianoKey)>) {
    let mut conn = MidiConn::new(2).unwrap();
    for (mut trans, &PianoKey { key, pitch }) in piano_keys.iter_mut() {
        let scale: &mut Vec3 = &mut trans.scale;
        let pitch = if keys.pressed(KeyCode::LAlt) {
            pitch - 12
        } else if keys.pressed(KeyCode::RAlt) {
            pitch + 12
        } else {
            pitch
        };
        if keys.just_pressed(key) {
            *scale *= 0.8;
            conn.play_on(pitch).unwrap();
        } else if keys.just_released(key) {
            *scale /= 0.8;
            conn.play_off(pitch).unwrap();
        }
    }
}

fn record_note(
    time: Res<Time>,
    keys: Res<Input<KeyCode>>,
    mut notes: ResMut<RecordNotes>,
    record: Res<RecordDuration>,
    mut piano_watch: Query<(&mut PlayDuration, &PianoKey)>,
) {
    for (mut duration, piano_key) in piano_watch.iter_mut() {
        if keys.just_pressed(piano_key.key) {
            duration.time.reset();
        } else if keys.pressed(piano_key.key) {
            duration.time.tick(time.delta());
        } else if keys.just_released(piano_key.key) {
            let note = Note {
                pitch: piano_key.pitch,
                step: record.time.elapsed_secs(),
                duration: duration.time.elapsed_secs(),
            };
            notes.0.push(note);
        }
    }
}

fn record_start(mut record: ResMut<RecordDuration>) {
    record.time.reset();
}

fn record_tick(time: Res<Time>, mut record: ResMut<RecordDuration>) {
    record.time.tick(time.delta());
}
