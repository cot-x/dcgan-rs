use tch::{nn, Tensor};

#[derive(Debug)]
pub struct Generator {
    net: nn::SequentialT
}

impl Generator {
    pub fn new(path: nn::Path, dim_in: i64) -> Generator {
            let dim_hidden = 512;
            let net = nn::seq_t()
                .add(nn::conv_transpose2d(&path / "ConvTranspose2d_1", dim_in, dim_hidden,
                                            4, nn::ConvTransposeConfig { stride: 1, padding: 0, dilation: 1, ..Default::default() }))
                .add(nn::batch_norm2d(&path / "batch_norm2d_1", dim_hidden, Default::default()))
                .add_fn(|x| x.relu())
                .add(nn::conv_transpose2d(&path / "ConvTranspose2d_2", dim_hidden, dim_hidden / 2,
                                            4, nn::ConvTransposeConfig { stride: 2, padding: 1, dilation: 1, ..Default::default() }))
                .add(nn::batch_norm2d(&path / "batch_norm2d_2", dim_hidden / 2, Default::default()))
                .add_fn(|x| x.relu())
                .add(nn::conv_transpose2d(&path / "ConvTranspose2d_3", dim_hidden / 2, dim_hidden / 4,
                                            4, nn::ConvTransposeConfig { stride: 2, padding: 1, dilation: 1, ..Default::default() }))
                .add(nn::batch_norm2d(&path / "batch_norm2d_3", dim_hidden / 4, Default::default()))
                .add_fn(|x| x.relu())
                .add(nn::conv_transpose2d(&path / "ConvTranspose2d_4", dim_hidden / 4, 3,
                                            4, nn::ConvTransposeConfig { stride: 2, padding: 1, dilation: 1, ..Default::default() }))
                .add_fn(|x| x.sigmoid());
                Generator { net }
        }
}

impl nn::ModuleT for Generator {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.net.forward_t(&xs, train)
        }
}
