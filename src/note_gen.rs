//! 神经网络预测音符模块

use std::path::Path;

use tch::{CModule, TchError, Tensor};

use crate::Note;

/// 包装神经网络模块
#[derive(Debug)]
pub struct TchMusicGenerator {
    model: CModule,
}

impl TchMusicGenerator {
    /// TODO use more flexible `path` parameter.
    /// 通过路径生成新的模型实例
    pub fn load_model() -> Result<TchMusicGenerator, TchError> {
        let pt_path = Path::new("./music_producter.pt");
        // TODO CUDA支持
        // 加载模型到Cpu
        tch::CModule::load_on_device(pt_path, tch::Device::Cpu).map(|model| Self { model })
    }

    /// 尝试用模型进行推理
    pub fn try_forward(&self, input: &[Note]) -> Result<tch::IValue, TchError> {
        // 将音符序列转换成输入用的张量
        let input = Tensor::of_slice2(
            input
                .iter()
                // TODO explain why pitch should div 128.0
                .map(|note| [note.pitch as f32 / 128.0, note.step, note.duration])
                .collect::<Vec<_>>()
                .as_slice(),
        );
        println!("input size is {:?}", &input.unsqueeze(0).size());

        // 推理并返回rust包装的pytorch类型
        self.model
            .forward_is(&[tch::IValue::Tensor(input.unsqueeze(0))])
    }

    /// 用模型进行推理
    pub fn forward(&self, input: &[Note]) -> Note {
        // 进行推理并把包装的pytorch类型转换为`Note`类型
        let (pitch, step, duration) = convert_iv_tuple(self.try_forward(input).unwrap()).unwrap();
        Note {
            pitch,
            step,
            duration,
        }
    }
}

/// 输出不符合要求的错误
#[derive(Debug)]
pub struct WrongValueError(tch::IValue);

/// 将rust包装的pytorch类型转换为rust的三元组类型
pub fn convert_iv_tuple(out: tch::IValue) -> Result<(u8, f32, f32), WrongValueError> {
    println!("output is : {:?}", &out);
    // 输出要是pytorch的元组类型且包含三个张量作为元素
    match out {
        tch::IValue::Tuple(v) => {
            if let [tch::IValue::Tensor(pitch), tch::IValue::Tensor(step), tch::IValue::Tensor(duration)] =
                &v[..]
            {
                // 第一个张量大小为1 x 128, 剩余俩个张量大小为 1 x 1
                if &pitch.size() == &[1, 128]
                    && &step.size() == &[1, 1]
                    && &duration.size() == &[1, 1]
                {
                    // 从张量中提取数据
                    let (pitch, step, duration) = (
                        pitch.squeeze(),
                        step.double_value(&[0, 0]) as f32,
                        duration.double_value(&[0, 0]) as f32,
                    );
                    // 将128个数据中最高值的索引作为音高
                    let (pitch, _) = pitch
                        .iter::<f64>()
                        .unwrap()
                        .enumerate()
                        .reduce(|pre, p| if pre.1 > p.1 { pre } else { p })
                        .unwrap();

                    // 将转换后的数据包装并返回
                    Ok((pitch as u8, step, duration))
                } else {
                    Err(WrongValueError(tch::IValue::Tuple(v)))
                }
            } else {
                Err(WrongValueError(tch::IValue::Tuple(v)))
            }
        }
        _ => Err(WrongValueError(out)),
    }
}

mod test {
    use tch::Tensor;

    use crate::note_gen::convert_iv_tuple;

    use super::TchMusicGenerator;
    use crate::Note;

    #[test]
    fn generator_test() {
        let gen = TchMusicGenerator::load_model().unwrap();
        let input = [
            Note {
                pitch: 57,
                step: 0.0,
                duration: 0.94895834,
            },
            Note {
                pitch: 69,
                step: 0.5000,
                duration: 0.4740,
            },
            Note {
                pitch: 48,
                step: 0.5000,
                duration: 0.9490,
            },
        ];

        let out = convert_iv_tuple(gen.try_forward(&input).unwrap());

        if let Ok(note) = out {
            println!("predict note : {:?}", note);
            assert!(true)
        } else {
            eprintln!("error output : {:?}", out);
            assert!(false)
        };
    }

    #[test]
    fn output_test() {
        let gen = TchMusicGenerator::load_model().unwrap();
        // let input = Tensor::of_slice(&[0.4453125f32, 0.0, 0.94895834]);
        let input = Tensor::of_slice2(&[
            [0.4453125f32, 0.0, 0.94895834],
            [0.5391, 0.5000, 0.4740],
            [0.3750, 0.5000, 0.9490],
        ]);
        println!("input size is {:?}", &input.unsqueeze(0).size());
        // gen.forward(input.unsqueeze(1)).unwrap().print();
        let out = gen
            .model
            .forward_is(&[tch::IValue::Tensor(input.unsqueeze(0))])
            .unwrap();
        eprintln!("output is {:?}", &out);
        if let tch::IValue::Tuple(v) = out {
            if let [tch::IValue::Tensor(pitch), tch::IValue::Tensor(step), tch::IValue::Tensor(duration)] =
                &v[..]
            {
                if &pitch.size() == &[1, 128]
                    && &step.size() == &[1, 1]
                    && &duration.size() == &[1, 1]
                {
                    let (pitch, step, duration) =
                        (pitch.squeeze(), step.squeeze(), duration.squeeze());
                    println!("predict note is {:?}", &(pitch.max(), step, duration));
                }
            }
        }
    }
}
