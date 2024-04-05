use tch::{nn, Tensor};

#[derive(Debug)]
pub struct Discriminator {
    net: nn::SequentialT
}

impl Discriminator {
    pub fn new(path: nn::Path) -> Discriminator {
            let dim_hidden = 128;
            let net = nn::seq_t()
                .add(nn::conv2d(&path / "Conv2d_1", 3, dim_hidden,
                                            4, nn::ConvConfig { stride: 2, padding: 1, ..Default::default() }))
                .add_fn(|x| x.leaky_relu())
                .add(nn::conv2d(&path / "Conv2d_2", dim_hidden, dim_hidden * 2,
                                            4, nn::ConvConfig { stride: 2, padding: 1, ..Default::default() }))
                .add(nn::batch_norm2d(&path / "batch_norm2d_2", dim_hidden * 2, Default::default()))
                .add_fn(|x| x.leaky_relu())
                .add(nn::conv2d(&path / "Conv2d_3", dim_hidden * 2, dim_hidden * 4,
                                            4, nn::ConvConfig { stride: 2, padding: 1, ..Default::default() }))
                .add(nn::batch_norm2d(&path / "batch_norm2d_3", dim_hidden * 4, Default::default()))
                .add_fn(|x| x.leaky_relu())
                .add(nn::conv2d(&path / "Conv2d_4", dim_hidden * 4, 1,
                                            3, nn::ConvConfig { stride: 1, padding: 1, ..Default::default() }));
            Discriminator { net }
        }
}

impl nn::ModuleT for Discriminator {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.net.forward_t(&xs, train)
        }
}
