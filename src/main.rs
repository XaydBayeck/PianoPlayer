mod midi;

use std::time::Duration;

use bevy::{core::Stopwatch, prelude::*, sprite::Anchor};
use midi::MidiConn;

const FALL_VECTOR: f32 = 200.0;

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
    let mut test_notes = TestNotes::from(vec![
        (87, 0., 0.5),
        (85, 0.5, 0.5),
        (87, 0.5, 0.5),
        (89, 0.5, 0.5),
        (90, 0.5, 1.),
        (87, 1., 0.5),
        (90, 0.5, 0.5),
        (89, 0.5, 0.5),
        (85, 0.5, 0.5),
        (85, 0.5, 0.5),
        (82, 0.5, 0.5),
        (85, 0.5, 2.),
        (83, 3., 1.),
        (85, 1., 1.),
        (83, 1., 1.),
        (82, 1., 0.5),
        (80, 0.5, 0.5),
        (82, 0.5, 0.5),
        (83, 0.5, 0.5),
        (85, 0.5, 1.),
        (82, 1., 1.),
        (87, 1., 0.5),
        (85, 0.5, 0.5),
        (87, 0.5, 0.5),
        (89, 0.5, 0.5),
        (90, 0.5, 1.),
        (87, 1., 0.5),
        (90, 0.5, 0.5),
        (89, 0.5, 0.5),
        (85, 0.5, 0.5),
        (85, 0.5, 0.5),
        (82, 0.5, 0.5),
        (85, 0.5, 2.),
        (83, 3., 1.),
        (87, 0., 1.),
        (89, 1., 1.),
        (85, 0., 1.),
        (85, 1., 1.),
        (80, 0., 1.),
        (82, 1., 0.5),
        (80, 0.5, 0.5),
        (82, 0.5, 0.5),
        (83, 0.5, 0.5),
        (85, 0.5, 1.),
        (82, 1., 1.),
        (87, 1., 0.5),
        (85, 0.5, 0.5),
        (87, 0.5, 0.5),
        (89, 0.5, 0.5),
        (90, 0.5, 1.),
        (87, 1., 0.5),
        (90, 0.5, 0.5),
        (89, 0.5, 0.5),
        (85, 0.5, 0.5),
        (85, 0.5, 0.5),
        (82, 0.5, 0.5),
        (85, 0.5, 2.),
        (83, 3., 1.),
        (87, 0., 1.),
        (89, 1., 1.),
        (85, 0., 1.),
        (85, 1., 1.),
        (80, 0., 1.),
        (82, 1., 0.5),
        (80, 0.5, 0.5),
        (82, 0.5, 0.5),
        (83, 0.5, 0.5),
        (85, 0.5, 1.),
        (82, 1., 1.),
        (80, 2., 1.),
        (82, 1., 1.),
        (83, 1., 1.),
        (82, 2., 1.),
        (83, 1., 1.),
        (85, 1., 1.),
        (83, 2., 1.),
        (85, 1., 1.),
        (87, 1., 1.),
        (89, 1., 4.),
        (90, 2., 2.),
        (92, 2., 12.),
    ]);
    test_notes.0.reverse();

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
        .insert_resource(test_notes)
        .insert_resource(PlaceTimer {
            timer: Timer::from_seconds(2.5, false),
        })
        .insert_resource(RecordDuration::default())
        .add_state(GameStates::FreePlaying)
        .add_startup_system(setup)
        .add_system_set(SystemSet::on_update(GameStates::FreePlaying).with_system(play_note))
        .add_system_set(SystemSet::on_enter(GameStates::Record).with_system(record_start))
        .add_system_set(
            SystemSet::on_update(GameStates::Record)
                .with_system(record_tick)
                .with_system(play_note)
                .with_system(record_note),
        )
        .add_system_set(
            SystemSet::on_update(GameStates::Playing)
                .with_system(test_place_notes)
                .with_system(note_fall)
                .with_system(detect_note),
        )
        .add_system(state_switch)
        // .add_system(play_note)
        .run();
}

// Resources
#[derive(Debug)]
struct RecordNotes(Vec<Note>);

#[derive(Debug, Default)]
struct RecordDuration {
    time: Stopwatch,
}

#[derive(Debug, Default)]
struct PlaceTimer {
    timer: Timer,
}

#[derive(Debug)]
struct TestNotes(Vec<Note>);

impl From<Vec<(u8, f32, f32)>> for TestNotes {
    fn from(notes: Vec<(u8, f32, f32)>) -> Self {
        Self(
            notes
                .into_iter()
                .map(|(pitch, step, duration)| Note {
                    pitch,
                    step,
                    duration,
                })
                .collect(),
        )
    }
}

// State
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum GameStates {
    Record,
    Playing,
    FreePlaying,
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

#[derive(Component, Debug)]
struct PlayPitch(u8);

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
                transform: Transform::from_xyz(x_position, 50.0 - HEIGHT / 2.0, 1.0),
                ..default()
            })
            .insert(PianoKey {
                key: *key,
                pitch: *pitch,
            })
            .insert(PlayPitch(*pitch))
            .insert(PlayDuration::default());
    }
}

fn state_switch(mut keys: ResMut<Input<KeyCode>>, mut state: ResMut<State<GameStates>>) {
    if keys.just_pressed(KeyCode::Tab) {
        match state.current() {
            GameStates::FreePlaying => {
                state.set(GameStates::Playing).unwrap();
                info!("State Switch to `Playing`!")
            }
            GameStates::Playing => {
                state.set(GameStates::FreePlaying).unwrap();
                info!("State Switch to `FreePlaying`!")
            }
            _ => (),
        }
        keys.reset(KeyCode::Q);
    } else if keys.just_pressed(KeyCode::Q) {
        match state.current() {
            GameStates::Record => {
                state.pop().unwrap();
                info!("Recording...")
            }
            _ => state.push(GameStates::Record).unwrap(),
        }
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

fn detect_note(
    keys: Res<Input<KeyCode>>,
    mut piano_keys: Query<(&mut Transform, &PianoKey, &mut PlayPitch), Without<Note>>,
    notes: Query<(&Transform, &Note), Without<PianoKey>>,
) {
    let mut conn = MidiConn::new(2).unwrap();
    for (mut trans, &PianoKey { key, pitch: _ }, mut p_pitch) in piano_keys.iter_mut() {
        let x_position = trans.translation.x;
        let scale: &mut Vec3 = &mut trans.scale;

        if keys.just_pressed(key) {
            for (n_trans, note) in notes.iter() {
                if n_trans.translation.y < (100.0 - HEIGHT as f32 / 2.0)
                    && n_trans.translation.x == x_position
                {
                    p_pitch.0 = note.pitch;
                    break;
                }
            }
            *scale *= 0.8;
            conn.play_on(p_pitch.0).unwrap();
        } else if keys.just_released(key) {
            *scale /= 0.8;
            conn.play_off(p_pitch.0).unwrap();
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
            info!("Hase record notes: {:?}", &notes.0);
        }
    }
}

fn record_start(mut record: ResMut<RecordDuration>) {
    record.time.reset();
}

fn record_tick(time: Res<Time>, mut record: ResMut<RecordDuration>) {
    record.time.tick(time.delta());
}

fn test_place_notes(
    mut commands: Commands,
    mut test_notes: ResMut<TestNotes>,
    time: Res<Time>,
    mut place_timer: ResMut<PlaceTimer>,
) {
    if place_timer.timer.just_finished() {
        if let Some(
            _n @ Note {
                pitch,
                step,
                duration,
            },
        ) = test_notes.0.pop()
        {
            // info!("get note {:?}", &_n);
            place_timer
                .timer
                .set_duration(Duration::from_secs_f32(duration));
            place_timer.timer.reset();

            let key_width = WIDTH / NOTES_TABLE.len() as f32;
            let length = FALL_VECTOR * duration;
            let x_position =
                key_width / 2.0 - WIDTH / 2.0 + reduce_octave(pitch) as f32 * key_width;

            commands
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        color: Color::AQUAMARINE,
                        custom_size: Some(Vec2::new(key_width - 10.0, length)),
                        anchor: Anchor::BottomCenter,
                        ..default()
                    },
                    transform: Transform::from_xyz(x_position, HEIGHT / 2.0, 0.0),
                    ..default()
                })
                .insert(Note {
                    pitch,
                    step,
                    duration,
                });
        }
    } else {
        place_timer.timer.tick(time.delta());
    }
}

fn reduce_octave(pitch: u8) -> u8 {
    ((pitch as i32 - 60) % 8).abs() as u8
}

fn note_fall(time: Res<Time>, mut notes: Query<&mut Transform, With<Note>>) {
    for mut trans in notes.iter_mut() {
        let pos: &mut Vec3 = &mut trans.translation;
        pos.y -= FALL_VECTOR * time.delta_seconds();
    }
}
