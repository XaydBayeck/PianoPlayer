mod midi;
mod note_gen;

use std::{collections::VecDeque, time::Duration};

use bevy::{core::Stopwatch, prelude::*, sprite::Anchor};
use midi::MidiConn;
use note_gen::TchMusicGenerator;

/// 游戏窗口的大小: 800 x 600
const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;

/// 音符下落的速度，琴键占100px,剩下高度为600-100=500px,
/// 下落速度设定为：200px/s
const FALL_VECTOR: f32 = 200.0;

/// 1 个八度（包含半音阶）的按键绑定
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

/// 程序的入口函数
fn main() {
    // 预存的曲谱
    let test_notes = TestNotes::from(VecDeque::from([
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
    ]));

    // 生成神经网络模型实例
    let model = TchMusicGenerator::load_model().unwrap();

    // Bevy的程序创建
    App::new()
        // 加载Bevy的默认插件
        // 包含图像渲染、事件循环等游戏所需的基本系统与内容
        .add_plugins(DefaultPlugins)
        // 插入窗口描述的资源
        .insert_resource(WindowDescriptor {
            width: WIDTH,
            height: HEIGHT,
            title: String::from("Piano Player"),
            present_mode: bevy::window::PresentMode::Fifo,
            resizable: false,
            ..default()
        })
        // 插入用于`Record`状态记录玩家输入的音符的数组资源
        .insert_resource(RecordNotes(vec![]))
        // 插入`Playing`状态缓存下落音符的队列（已经预存有音符）资源
        .insert_resource(test_notes)
        // 插入神经网络实例资源
        .insert_resource(model)
        // 插入控制音符下落的定时器资源
        .insert_resource(PlaceTimer {
            timer: Timer::from_seconds(2.5, false),
        })
        // 插入`Record`状态下使用的计时器资源
        .insert_resource(RecordDuration::default())
        // 插入`GameStates`状态资源, 以`FreePlaying`状态作为起始状态
        .add_state(GameStates::FreePlaying)
        // 创建2D摄像机、生成琴键
        .add_startup_system(setup)
        // 生成UI（指示状态的文字）
        .add_startup_system(setup_ui)
        // 在`FreePlaying`状态下运行的函数集
        // 包含一个普通无检测的钢琴模拟系统
        .add_system_set(SystemSet::on_update(GameStates::FreePlaying).with_system(play_note))
        // 在进入`Record`状态时的函数集
        .add_system_set(SystemSet::on_enter(GameStates::Record).with_system(record_start))
        // 在离开`Record`状态时的函数集
        .add_system_set(SystemSet::on_exit(GameStates::Record).with_system(record_over))
        // 在`Record`状态时的函数集
        // 包含无检测的钢琴模拟系统，输入记录系统，计时器系统
        .add_system_set(
            SystemSet::on_update(GameStates::Record)
                .with_system(record_tick)
                .with_system(play_note)
                .with_system(record_note),
        )
        // 在`Plying`状态下的函数集
        // 包含音符预测系统、音符生成系统、音符下落系统、有检测的钢琴模拟系统
        .add_system_set(
            SystemSet::on_update(GameStates::Playing)
                .with_system(model_gen_note.before(test_place_notes))
                .with_system(test_place_notes)
                .with_system(note_fall)
                .with_system(detect_note),
        )
        // 注册运行在整个游戏生命周期的系统
        .add_system(state_switch)
        .add_system(display_state)
        // 运行游戏
        .run();
}

// Resources
// 游戏资源是游戏中全局管理的数据

/// `Record`状态下记录玩家输入的数组包装结构
#[derive(Debug)]
struct RecordNotes(Vec<Note>);

/// 记录`Record`状态时间经过的计时器
#[derive(Debug, Default)]
struct RecordDuration {
    time: Stopwatch,
}

/// `Plying`状态下控制音符下落的定时器
#[derive(Debug, Default)]
struct PlaceTimer {
    timer: Timer,
}

/// 缓存下落音符的队列
#[derive(Debug)]
struct TestNotes(VecDeque<Note>);

/// 将包含元组类型`(u8, f32, f32)`的队列转换为包含`Note`类型的队列
impl From<VecDeque<(u8, f32, f32)>> for TestNotes {
    fn from(notes: VecDeque<(u8, f32, f32)>) -> Self {
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
// 游戏的状态，使得游戏在不同状态间转换，控制不同系统集的运行与停止

/// 游戏主要的状态枚举
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum GameStates {
    Record,
    Playing,
    FreePlaying,
}

// Components
// 游戏的组件，由游戏实体拥有的数据，决定实体具有那些行为

/// 包含音符数据的结构体
#[derive(Component, Debug, Clone, Copy)]
pub struct Note {
    /// 音高
    pitch: u8,
    // velocity:
    /// 间隔
    step: f32,
    /// 时值
    duration: f32,
}

/// `Record`状态下记录用户输入的音符的时值
#[derive(Component, Default, Debug)]
struct PlayDuration {
    time: Stopwatch,
}

/// 包含钢琴琴键数据的结构体
#[derive(Component, Debug)]
struct PianoKey {
    /// 绑定的按键
    key: KeyCode,
    /// 琴键的基础音高
    pitch: u8,
}

/// `Playing`状态下琴键实际演奏的音符音高
#[derive(Component, Debug)]
struct PlayPitch(u8);

/// 标识状态板实体的无数据组件
#[derive(Component, Debug)]
struct StateBoard;

// Systems
// 系统是游戏运行时每帧会对游戏资源、实体的组件数据进行处理的函数

/// 初始化函数生成摄像机和琴键
fn setup(mut commands: Commands) {
    // 生成2D的摄像机
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
    // 通过窗口宽度计算琴键的宽度
    let key_width = WIDTH / NOTES_TABLE.len() as f32;
    // 通过预先绑定的琴键数据生成琴键
    for (idx, (key, pitch, color)) in NOTES_TABLE.iter().enumerate() {
        // 计算琴键在水平方向的位置
        let x_position = key_width / 2.0 - WIDTH / 2.0 + idx as f32 * key_width;
        // 生成琴键实体
        commands
            // 通过精灵图组件集生成实体
            .spawn_bundle(SpriteBundle {
                // 精灵图组件
                sprite: Sprite {
                    color: *color,
                    custom_size: Some(Vec2::new(key_width - 2.0, 100.0)),
                    ..default()
                },
                // 局部变换组件（无父实体，局部变换与全局变换相同）
                transform: Transform::from_xyz(x_position, 50.0 - HEIGHT / 2.0, 1.0),
                // 使用默认值初始化剩余组件
                ..default()
            })
            // 插入琴键组件
            .insert(PianoKey {
                key: *key,
                pitch: *pitch,
            })
            // 插入演奏时音高组件
            .insert(PlayPitch(*pitch))
            // 插入时值记录组件
            .insert(PlayDuration::default());
    }
}


/// UI初始化系统
fn setup_ui(mut commands: Commands, asset_sever: Res<AssetServer>) {
    // 加载字体资源
    let font = asset_sever.load("fonts/QuinzeNarrow.ttf");
    // 设置字体样式
    let text_style = TextStyle {
        font,
        font_size: 60.0,
        color: Color::BLACK,
    };

    // 生成标识状态的UI实体
    commands
        // 通过文字组件集生成实体
        .spawn_bundle(Text2dBundle {
            // 文本内容组件
            text: Text::with_section(
                format!("Score:{:}", 0),
                text_style,
                TextAlignment {
                    vertical: VerticalAlign::Center,
                    horizontal: HorizontalAlign::Left,
                },
            ),
            // 局部变换组件
            transform: Transform::from_xyz(50.0 - 400.0, 300.0 - 50.0, 2.0),
            // 默认值初始化剩余组件
            ..default()
        })
        // 插入状态板组件
        .insert(StateBoard);
}

/// 显示当前状态
/// parameters:
///     - state : 游戏的全局状态资源
///     - ui_texts: 查询具有`StateBoard`和`Text`组件的实体的`Text`组件的可变引用
fn display_state(state: Res<State<GameStates>>, mut ui_texts: Query<&mut Text, With<StateBoard>>) {
    // 处理所有满足条件的实体的`Text`组件
    for mut ui_text in ui_texts.iter_mut() {
        // 判断游戏的全局状态是否在上一帧发生改变
        if state.is_changed() {
            // 根据当前的状态修改该组件的文本内容
            ui_text.sections[0].value = String::from(match state.current() {
                GameStates::Record => "Record",
                GameStates::Playing => "Playing",
                GameStates::FreePlaying => "FreePlaying",
            });
        }
    }
}

/// 游戏状态切换
/// parameters:
///     - keys: 管理游戏的全局输入的资源的可变引用
///     - state： 游戏的全局状态资源的可变引用
fn state_switch(mut keys: ResMut<Input<KeyCode>>, mut state: ResMut<State<GameStates>>) {
    // 检测`Tab`键是否被按下
    if keys.just_pressed(KeyCode::Tab) {
        // 在`FreePlaying`与`Playing`状态间进行切换
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
        // 重设按键避免潜在的bug
        keys.reset(KeyCode::Tab);
    // 检测`Q`键是否被按下
    } else if keys.just_pressed(KeyCode::Q) {
        // 将`Record`状态推入或推出栈中
        match state.current() {
            GameStates::Record => {
                state.pop().unwrap();
                info!("Recording...")
            }
            _ => state.push(GameStates::Record).unwrap(),
        }
        // 重设按键避免潜在的bug
        keys.reset(KeyCode::Q);
    }
}

/// 无检测的琴键模拟系统
/// parameters:
///     - keys: 管理全局输入的资源
///     - piano_keys: 查询`Transform`和`PianoKey`组件的部分可变引用
fn play_note(keys: Res<Input<KeyCode>>, mut piano_keys: Query<(&mut Transform, &PianoKey)>) {
    // 连接midi端口
    let mut conn = MidiConn::new(1).unwrap();
    // 处理符合条件的实体的被查询组件
    for (mut trans, &PianoKey { key, pitch }) in piano_keys.iter_mut() {
        // 绑定局部变化组件的缩放量
        let scale: &mut Vec3 = &mut trans.scale;
        // 按下左右`Alt`键分别控制音符降低或提升1个8度
        let pitch = if keys.pressed(KeyCode::LAlt) {
            pitch - 12
        } else if keys.pressed(KeyCode::RAlt) {
            pitch + 12
        } else {
            pitch
        };
        // 检测模拟琴键绑定的键盘按键
        // 按下或释放按键时分别发送打开或关闭音符的信息
        if keys.just_pressed(key) {
            // 按下琴键时对琴键缩放
            *scale *= 0.8;
            conn.play_on(pitch).unwrap();
        } else if keys.just_released(key) {
            // 恢复琴键大小
            *scale /= 0.8;
            conn.play_off(pitch).unwrap();
        }
    }
}

/// 有检测的琴键模拟系统
/// parameters:
///     - keys: 管理全局输入的资源
///     - piano_keys: 查询`Transform`、`PianoKey`和`PlayPitch`组件的部分可变引用
///     - notes: 查询`Transform`和`Note`组件的不可变引用
fn detect_note(
    keys: Res<Input<KeyCode>>,
    mut piano_keys: Query<(&mut Transform, &PianoKey, &mut PlayPitch), Without<Note>>,
    notes: Query<(&Transform, &Note), Without<PianoKey>>,
) {
    // 连接midi端口
    let mut conn = MidiConn::new(1).unwrap();
    // 处理符合条件的实体
    for (mut trans, &PianoKey { key, pitch }, mut p_pitch) in piano_keys.iter_mut() {
        // 绑定琴键水平方向坐标
        let x_position = trans.translation.x;
        // 绑定琴键的局部变化的缩放量
        let scale: &mut Vec3 = &mut trans.scale;

        // 检测琴键对应的按键输入
        // 按下或释放的时候分别发送音符的播放或停止信息
        if keys.just_pressed(key) {
            // 检查具有`Note`组件的实体的水平坐标是否与琴键重叠
            // 同时音符是否下落到琴键位置
            for (n_trans, note) in notes.iter() {
                if n_trans.translation.y < (100.0 - HEIGHT as f32 / 2.0)
                    && n_trans.translation.x == x_position
                {
                    // 将琴键的演奏音高修改为符合条件的下落音符的音高
                    p_pitch.0 = note.pitch;
                    info!("p_pitch has change to {:?}", p_pitch.0);
                    break;
                }
            }
            *scale *= 0.8;
            conn.play_on(p_pitch.0).unwrap();
        } else if keys.just_released(key) {
            *scale /= 0.8;
            conn.play_off(p_pitch.0).unwrap();
            // 释放按键时恢复原本音高
            p_pitch.0 = pitch;
            info!("p_pitch has change back to {:?}", p_pitch.0);
        }
    }
}

/// 记录玩家输入的音符
/// parameters:
///     - time: 系统时间资源
///     - keys: 全局输入资源
///     - record: 记录`Record`状态时间的计时器
///     - piano_watch: 查询`PiayDuraiton`和`PianoKey`组件的部分可变引用
fn record_note(
    time: Res<Time>,
    keys: Res<Input<KeyCode>>,
    mut notes: ResMut<RecordNotes>,
    record: Res<RecordDuration>,
    mut piano_watch: Query<(&mut PlayDuration, &PianoKey)>,
) {
    // 对符合条件的实体的组件进行处理
    for (mut duration, piano_key) in piano_watch.iter_mut() {
        // 检测琴键是否按下或释放
        if keys.just_pressed(piano_key.key) {
            // 重设音符时值计时器
            duration.time.reset();
        } else if keys.pressed(piano_key.key) {
            // 音符时值计时器计时
            duration.time.tick(time.delta());
        } else if keys.just_released(piano_key.key) {
            // 记录输入音符
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

/// `Record`状态的初始化系统
/// parameters：
///     - record: 记录`Record`状态时间的计时器
fn record_start(mut record: ResMut<RecordDuration>) {
    // 启动计时器
    if record.time.paused() {
        record.time.unpause();
    }
    // 计时器时间清零
    record.time.reset();
}

/// `Record`状态计时系统
/// parameters:
///     - time: 系统时间资源
///     - record: 记录`Record`状态时间的计时器
fn record_tick(time: Res<Time>, mut record: ResMut<RecordDuration>) {
    // 计时器计时
    record.time.tick(time.delta());
}

/// 离开`Record`状态时处理
/// parameters：
///     - record: 记录`Record`状态时间的计时器
///     - record_note: 记录玩家输入的音符数组
///     - test_notes: 缓存下落音符的队列
///     - modle: 预测音符的神经网络模型实例资源
fn record_over(
    mut record: ResMut<RecordDuration>,
    mut record_note: ResMut<RecordNotes>,
    mut test_notes: ResMut<TestNotes>,
    modle: Res<TchMusicGenerator>,
) {
    // 停止记录`Record`状态时间的计时器
    record.time.pause();
    // 将记录的音符加入下落音符缓存缓存队列
    while let Some(note) = record_note.0.pop() {
        test_notes.0.push_front(note);
    }
    // 利用队列中现有的音符预测下一个音符并加入队列
    let pred_note = modle.forward(test_notes.0.as_slices().0);
    test_notes.0.push_back(pred_note);
}

/// 音符预测系统
/// parameters:
///     - test_notes: 缓存下落音符的队列
///     - modle: 预测音符的神经网络模型实例资源
fn model_gen_note(mut test_notes: ResMut<TestNotes>, modle: Res<TchMusicGenerator>) {
    // 判断下落音符缓存队列中音符数量是否小于50个
    if test_notes.0.len() <= 50 {
        // 预测音符并加入队列
        let pred_note = modle.forward(test_notes.0.as_slices().0);
        test_notes.0.push_back(pred_note);
    }
}

/// 下落音符放置系统
/// parameters:
///     - commands: 控制世界行为的特殊输入
///     - test_notes: 缓存下落音符的队列
///     - time: 系统时间资源
///     - place_timer: 控制下落音符生成的定时器
fn test_place_notes(
    mut commands: Commands,
    mut test_notes: ResMut<TestNotes>,
    time: Res<Time>,
    mut place_timer: ResMut<PlaceTimer>,
) {
    // 检查定时器是否计时完成
    if place_timer.timer.just_finished() {
        // 从队列顶中取出一个音符
        if let Some(
            _n @ Note {
                pitch,
                step,
                duration,
            },
        ) = test_notes.0.pop_front()
        {
            // info!("get note {:?}", &_n);
            // 重新设定定时器的时间
            place_timer
                .timer
                .set_duration(Duration::from_secs_f32(duration));
            place_timer.timer.reset();

            // 计算音符宽度
            let key_width = WIDTH / NOTES_TABLE.len() as f32;
            // 根据时值计算音符长度（长度 = 下落速度 x 时值）
            let length = FALL_VECTOR * duration;
            // 计算音符的水平坐标
            //（超过琴键的基本音高的音符计算其对应的音名放在同音名的琴键水平坐标处）
            let x_position =
                key_width / 2.0 - WIDTH / 2.0 + reduce_octave(pitch) as f32 * key_width;

            commands
                // 通过精灵图组件集生成实体
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
                // 插入音符组件
                .insert(Note {
                    pitch,
                    step,
                    duration,
                });
        }
    } else {
        // 定时器计时
        place_timer.timer.tick(time.delta());
    }
}

/// 获得与音高同音名的琴键顺序的辅助函数
fn reduce_octave(pitch: u8) -> u8 {
    ((pitch as i32 - 60) % 8).abs() as u8
}


/// 音符下落系统
/// parameters:
///     - commands: 控制世界行为的特殊输入
///     - time: 系统时间资源
///     - notes: 查询具有`Note`和`Transform`组件的实体的实体ID和`Transform`组件可变引用
fn note_fall(
    mut commands: Commands,
    time: Res<Time>,
    mut notes: Query<(Entity, &mut Transform), With<Note>>,
) {
    // 对满足条件的实体的组件进行处理
    for (entity, mut trans) in notes.iter_mut() {
        // 按照设定的速度减少纵坐标
        let pos: &mut Vec3 = &mut trans.translation;
        pos.y -= FALL_VECTOR * time.delta_seconds();
        
        // TODO 更改为音符尾纵坐标低于琴键头部纵坐标
        // 检查音符是否掉落到窗口外
        if pos.y < -HEIGHT / 2.0 {
            // 删除音符实体
            commands.entity(entity).despawn()
        }
    }
}
