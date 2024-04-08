use std::path::Path;
use std::fs;
use std::fs::File;

use tch::{nn, nn::OptimizerConfig, Tensor, Device, kind, Kind};
use anyhow::{bail, Result};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use tqdm::tqdm;
use clap::Parser;
use gif::{Frame, Encoder, Repeat};

mod generator;
mod discriminator;
use generator::Generator;
use discriminator::Discriminator;

const IMG_SIZE: i64 = 64;
const DIM_LATENT: i64 = 64;

#[derive(Parser, Debug)]
#[command(version, about = "DCGAN via Rust.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = String::new())]
    dataset: String,
    #[arg(long, default_value_t = 32)]
    batch_size: i64,
    #[arg(long, default_value_t = 0.0001)]
    lr: f64,
    #[arg(long, default_value_t = 4.0)]
    mul_dis_lr: f64,
    #[arg(long, default_value_t = 0.6)]
    aug_threshold: f64,
    #[arg(long, default_value_t = 0.01)]
    aug_increment: f64,
    #[arg(short, long, default_value_t = 10000)]
    iters: i64,
    #[arg(short, long, default_value_t = 0)]
    generate: usize,
    #[arg(short, long, default_value_t = false)]
    morphing: bool
}

fn main() -> Result<()> {
    let args = Args::parse();

    print_oss_lisence();

    let device = Device::cuda_if_available();
    println!("Use Device: {device:?}");
    println!("--------");

    // Build Generator
    let mut g_vs = nn::VarStore::new(device);
    let generator = Generator::new(g_vs.root(), DIM_LATENT);
    let mut optimizer_g = nn::adam(0., 0.9, 0.).build(&g_vs, args.lr)?;

    // Build Discriminator
    let mut d_vs = nn::VarStore::new(device);
    let discriminator = Discriminator::new(d_vs.root());
    let mut optimizer_d = nn::adam(0., 0.9, 0.).build(&d_vs, args.lr * args.mul_dis_lr)?;

    // Load network's weights
    let path = Path::new("g_weights.pth");
    if path.is_file() == true {
        g_vs.load("g_weights.pth")?;
        println!("Loaded: g_weights.pth");
    }
    let path = Path::new("d_weights.pth");
    if path.is_file() == true {
        d_vs.load("d_weights.pth")?;
        println!("Loaded: d_weights.pth");
    }

    if args.generate != 0 {
        generate(&generator, args.generate, device);
        println!("Generated!");
        return Ok(());
    }

    if args.morphing {
        morphing(&generator, device);
        println!("Generated morphing.gif");
        return Ok(());
    }

    if args.dataset == "" {
        bail!("Please specify the dataset. See the --help option for details.")
    }

    println!("Image Dataset Dir: {}", args.dataset);
    println!("Image Size: {}", IMG_SIZE);
    println!("Dim Latent: {}", DIM_LATENT);
    println!("Batch Size: {}", args.batch_size);
    println!("Learning Rate: {}", args.lr);
    println!("Mul Discriminator's LR: {}", args.mul_dis_lr);
    println!("Probability Aut-Threshold: {}", args.aug_threshold);
    println!("Probability Aug-Increment: {}", args.aug_increment);
    println!("Iterations: {}", args.iters);
    println!("--------");

    println!("Loading Dataset...");
    let dataset = tch::vision::image::load_dir(args.dataset, IMG_SIZE, IMG_SIZE)?;
    println!("Loaded Dataset: {dataset:?}");
    let train_size = dataset.size()[0];
    println!("--------");

    // Train with LSGAN.
    // for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
    let a = 0;
    let b = 1;
    let c = 1;

    let minibatch = || {
        let index = Tensor::randint(train_size, [args.batch_size], kind::INT64_CPU);
        dataset.index_select(0, &index).to_device(device).to_kind(Kind::Float) / 255.0
    };

    let seed = || {
        Tensor::rand([args.batch_size, DIM_LATENT, 1, 1], kind::FLOAT_CPU).to_device(device)
    };

    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);

    // for APA
    let mut pseudo_aug: f64 = 0.;

    println!("Train Iteration...");
    for i in tqdm(0..args.iters) {
        // Train Discriminator
        d_vs.unfreeze();
        g_vs.freeze();
        let d_loss = {
            let batch = minibatch();

            let real_src_score = batch.apply_t(&discriminator, true);
            let real_src_loss = (&real_src_score - b).square().sum(Kind::Float);

            let fake_src_score = seed()
                .apply_t(&generator, true)
                .copy()
                .detach()
                .apply_t(&discriminator, true);

            // APA
            let fake_src_loss: Tensor;
            let p = rng.sample(uniform);
            if 1. - pseudo_aug < p {
                fake_src_loss = (&fake_src_score - b).square().sum(Kind::Float); // Pseudo: fake is real.
            } else {
                fake_src_loss = (&fake_src_score - a).square().sum(Kind::Float);
            }

            // Update Probability Augmentation
            let lz = (real_src_score.logit(None).sign().mean(Kind::Float) - fake_src_score.logit(None).sign().mean(Kind::Float)) / 2;
            if lz.unsqueeze(0).iter::<f64>()?.next().unwrap() > args.aug_threshold {
                pseudo_aug += args.aug_increment;
            } else {
                pseudo_aug -= args.aug_increment;
            }
            pseudo_aug = f64::min(1., f64::max(0., pseudo_aug));

            0.5 * (real_src_loss + fake_src_loss) / args.batch_size
        };
        optimizer_d.backward_step(&d_loss);

        // Train Generator
        d_vs.freeze();
        g_vs.unfreeze();
        let g_loss = {
            let fake_src_score = seed()
                .apply_t(&generator, true)
                .apply_t(&discriminator, true);
            let fake_src_loss = (&fake_src_score - c).square().sum(Kind::Float);

            0.5 * fake_src_loss / args.batch_size
        };
        optimizer_g.backward_step(&g_loss);

        // Output
        if (i+1) % 100 == 0 {
            println!("{}: G_LOSS({:?}) + D_LOSS({:?}) = {:?}", i+1, &g_loss, &d_loss, &g_loss + &d_loss);

            g_vs.save("g_weights.pth")?;
            d_vs.save("d_weights.pth")?;

            let seed_1 = || {
                Tensor::rand([1, DIM_LATENT, 1, 1], kind::FLOAT_CPU).to_device(device)
            };
            let image = seed_1()
                .apply_t(&generator, false)
                .squeeze_dim(0) * 255.0;
            let path = Path::new("results");
            if path.is_dir() == false {
                fs::create_dir("results")?;
            }
            let mut path_buf = path.to_path_buf();
            path_buf.push(format!("{}.png", i+1));
            tch::vision::image::save(&image, &path_buf)?
        }
    }

    Ok(())
}

fn generate(generator: &Generator, num: usize, device: Device) {
    for i in 0..num {
        let seed_1 = || {
            Tensor::rand([1, DIM_LATENT, 1, 1], kind::FLOAT_CPU).to_device(device)
        };
        let image = seed_1()
            .apply_t(generator, false)
            .squeeze_dim(0) * 255.0;
        let path = Path::new("results");
        if path.is_dir() == false {
            fs::create_dir("results").unwrap();
        }
        let mut path_buf = path.to_path_buf();
        path_buf.push(format!("generate-{}.png", i+1));
        tch::vision::image::save(&image, &path_buf).unwrap();
    }
}

fn morphing(generator: &Generator, device: Device) {
    let seed_one = Tensor::from_slice(&[0f32; DIM_LATENT as usize]).reshape([1, -1, 1, 1]).to_device(device);
    let seed_two = Tensor::from_slice(&[1f32; DIM_LATENT as usize]).reshape([1, -1, 1, 1]).to_device(device);

    let path = Path::new("results");
    if path.is_dir() == false {
        fs::create_dir("results").unwrap();
    }
    let mut path_buf = path.to_path_buf();
    path_buf.push("morphing.gif");
    let mut image = File::create(path_buf).unwrap();
    let mut encoder = Encoder::new(&mut image, IMG_SIZE as u16, IMG_SIZE as u16, &[]).unwrap();
    encoder.set_repeat(Repeat::Infinite).unwrap();

    for i in 0..100 {
        let l:f64 = i as f64 / 99.0;
        let image = ((1.0 - l) * &seed_one + l * &seed_two)
            .apply_t(generator, false)
            .squeeze_dim(0).permute_copy([1,2,0]).flat_view() * 255.0;
        let mut data = [0 as u8; 3 * IMG_SIZE as usize * IMG_SIZE as usize];
        image.to_kind(Kind::Uint8).copy_data(&mut data, 3 * IMG_SIZE as usize * IMG_SIZE as usize);
        let frame = Frame::from_rgb(IMG_SIZE as u16, IMG_SIZE as u16, &data);
        encoder.write_frame(&frame).unwrap();
    }
}

fn print_oss_lisence() {
    println!("OSS Library: PyTorch");
    println!("URL: https://github.com/pytorch/pytorch");
    println!("Lisence: 3-Clause BSD License");
    println!("");
    println!("OSS Library: tch-rs");
    println!("URL: https://github.com/LaurentMazare/tch-rs");
    println!("Lisence: MIT Lisence");
    println!("");
    println!("OSS Library: anyhow");
    println!("URL: https://github.com/dtolnay/anyhow");
    println!("Lisence: Apache License Version 2.0");
    println!("");
    println!("OSS Library: rand");
    println!("URL: https://github.com/rust-random/rand");
    println!("Lisence: MIT OR Apache-2.0");
    println!("");
    println!("OSS Library: gif");
    println!("URL: https://github.com/image-rs/image-gif");
    println!("Lisence: MIT OR Apache-2.0");
    println!("");
    println!("OSS Library: tqdm");
    println!("URL: https://github.com/mrlazy1708/tqdm");
    println!("Lisence: MIT OR Apache-2.0");
    println!("");
    println!("OSS Library: clap");
    println!("URL: https://github.com/clap-rs/clap");
    println!("Lisence: MIT OR Apache-2.0");
    println!("--------");
    println!("");
}
