use std::{borrow::Borrow, path::Path};

use tch::{nn::Module, CModule, TchError, Tensor};

use crate::Note;

#[derive(Debug)]
pub struct TchMusicGenerator {
    model: CModule,
}

impl TchMusicGenerator {
    // TODO use more flexible `path` parameter.
    pub fn load_model() -> Result<TchMusicGenerator, TchError> {
        let pt_path = Path::new("./music_producter.pt");
        tch::CModule::load_on_device(pt_path, tch::Device::Cpu).map(|model| Self { model })
    }

    // pub fn forward(&self, input: Tensor) -> Result<tch::Tensor, TchError> {
    pub fn try_forward(&self, input: &[Note]) -> Result<tch::IValue, TchError> {
        let input = Tensor::of_slice2(
            input
                .iter()
                // TODO explain why pitch should div 128.0
                .map(|note| [note.pitch as f32 / 128.0, note.step, note.duration])
                .collect::<Vec<_>>()
                .as_slice(),
        );
        println!(
            "input size is {:?}",
            &input.unsqueeze(0).size()
        );
        self.model.forward_is(&[tch::IValue::Tensor(input.unsqueeze(0))])
    }

    pub fn forward(&self, input: &[Note]) -> Note {
        let (pitch, step, duration) = convert_iv_tuple(self.try_forward(input).unwrap()).unwrap();
        Note {
            pitch,
            step,
            duration,
        }
    }
}

#[derive(Debug)]
pub struct WrongValueError(tch::IValue);

pub fn convert_iv_tuple(out: tch::IValue) -> Result<(u8, f32, f32), WrongValueError> {
    println!("output is : {:?}", &out);
    match out {
        tch::IValue::Tuple(v) => {
            if let [tch::IValue::Tensor(pitch), tch::IValue::Tensor(step), tch::IValue::Tensor(duration)] =
                &v[..]
            {
                if &pitch.size() == &[1, 128]
                    && &step.size() == &[1, 1]
                    && &duration.size() == &[1, 1]
                {
                    let (pitch, step, duration) = (
                        pitch.squeeze(),
                        step.double_value(&[0, 0]) as f32,
                        duration.double_value(&[0, 0]) as f32,
                    );
                    let (pitch, _) = pitch
                        .iter::<f64>()
                        .unwrap()
                        .enumerate()
                        .reduce(|pre, p| if pre.1 > p.1 { pre } else { p })
                        .unwrap();

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
        println!(
            "input size is {:?}",
            &input.unsqueeze(0).size()
        );
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
                    // println!("predict note is {:?}", &(pitch.max(), step, duration));
                }
            }
        }
    }
}
