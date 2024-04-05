use std::path::Path;
use anyhow::{bail, Result};
use tch::{nn, nn::OptimizerConfig, Tensor, Device, kind, Kind};

mod generator;
mod discriminator;
use generator::Generator;
use discriminator::Discriminator;

const IMG_SIZE: i64 = 32;
const DIM_LATENT: i64 = 32;
const BATCH_SIZE: i64 = 32;
const LEARNING_RATE: f64 = 0.0001;
const MUL_LR_DIS: f64 = 4.;
const BATCHES: i64 = 100000000;

fn main() -> Result<()> {
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
    println!("--------");
    println!("");

    let args: Vec<_> = std::env::args().collect();
    let image_dir = match &args[..] {
        [_, dir] => dir,
        _ => bail!("Usage: main.exe image-dataset-dir")
    };
    println!("Image Dataset Dir: {image_dir}");

    let device = Device::cuda_if_available();
    println!("Use Device: {device:?}");

    println!("Image Size: {IMG_SIZE}");
    println!("Dim Latent: {DIM_LATENT}");
    println!("Batch Size: {BATCH_SIZE}");
    println!("Learning Rate: {LEARNING_RATE}");
    println!("Mul Discriminator's LR: {MUL_LR_DIS}");
    println!("BATCHES: {BATCHES}");
    println!("--------");

    println!("Loading Dataset...");
    let dataset = tch::vision::image::load_dir(image_dir, IMG_SIZE, IMG_SIZE)?;
    println!("Loaded Dataset: {dataset:?}");
    let train_size = dataset.size()[0];
    println!("Train Size: {}", train_size);
    println!("--------");

    let mut g_vs = nn::VarStore::new(device);
    let generator = Generator::new(g_vs.root(), DIM_LATENT);
    let mut optimizer_g = nn::adam(0., 0.9, 0.).build(&g_vs, LEARNING_RATE)?;

    let mut d_vs = nn::VarStore::new(device);
    let discriminator = Discriminator::new(d_vs.root());
    let mut optimizer_d = nn::adam(0., 0.9, 0.).build(&d_vs, LEARNING_RATE * MUL_LR_DIS)?;

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

    let minibatch = || {
        let index = Tensor::randint(train_size, [BATCH_SIZE], kind::INT64_CPU);
        dataset.index_select(0, &index).to_device(device).to_kind(Kind::Float) / 255.0
    };

    let seed = || {
        Tensor::rand([BATCH_SIZE, DIM_LATENT, 1, 1], kind::FLOAT_CPU).to_device(device)
    };

    // Train with LSGAN.
    // for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
    let a = 0;
    let b = 1;
    let c = 1;

    for i in 0..BATCHES {
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
            let fake_src_loss = (&fake_src_score - a).square().sum(Kind::Float);

            0.5 * (real_src_loss + fake_src_loss) / BATCH_SIZE
        };
        optimizer_d.backward_step(&d_loss);

        d_vs.freeze();
        g_vs.unfreeze();
        let g_loss = {
            let fake_src_score = seed()
                .apply_t(&generator, true)
                .apply_t(&discriminator, true);
            let fake_src_loss = (&fake_src_score - c).square().sum(Kind::Float);

            0.5 * fake_src_loss / BATCH_SIZE
        };
        optimizer_g.backward_step(&g_loss);

        print!("*");
        if (i+1) % 100 == 0 {
            println!("");
            println!("{}: G_LOSS({:?}) + D_LOSS({:?}) = {:?}", i+1, &g_loss, &d_loss, &g_loss + &d_loss);

            g_vs.save("g_weights.pth")?;
            d_vs.save("d_weights.pth")?;

            let seed_1 = || {
                Tensor::rand([1, DIM_LATENT, 1, 1], kind::FLOAT_CPU).to_device(device)
            };
            let image = seed_1()
                .apply_t(&generator, false)
                .squeeze_dim(0);
            tch::vision::image::save(&image, format!("{}.png", i+1))?
        }
    }

    Ok(())
}
