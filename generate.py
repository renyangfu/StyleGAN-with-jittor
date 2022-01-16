import argparse
import math
import numpy as np
import jittor as jt
from jittor import init
from jittor import nn

from model import StyledGenerator, Discriminator
#jt.flags.use_cuda = 1

def get_mean_style(generator):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(jt.init.gauss((1024, 512), 'float32'))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


def sample(generator, step, mean_style, n_sample):
    image = generator(
        jt.init.gauss((n_sample, 512), 'float32'),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )

    return image


def style_mixing(generator, step, mean_style, n_source, n_target):
    source_code = jt.init.gauss((n_source, 512), 'float32')
    target_code = jt.init.gauss((n_target, 512), 'float32')

    shape = 4 * 2 ** step
    alpha = 1

    images = [jt.ones((1, 3, shape, shape), 'float32') * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = jt.concat(images, 0)

    return images


def latentinterpolation(generator, step, mean_style, nrow, ncol):
    source_code1, source_code2 = jt.init.gauss((2, 512), 'float32').chunk(2, 0)
    target_code1, target_code2 = jt.init.gauss((2, 512), 'float32').chunk(2, 0)

    shape = 4 * 2 ** step
    alpha = 1

    #images = [jt.ones((1, 3, shape, shape), 'float32') * -1]
    images = []
    r = nrow-1
    c = ncol-1
    for i in range(nrow):
        s = source_code1*((r-i)/r)+source_code2*(i/r)
        t = target_code1*((r-i)/r)+target_code2*(i/r)
        for k in range(ncol):
            code = s*((c-k)/c)+t*(k/c)
            interp_img = generator(code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)
            images.append(interp_img)

    images = jt.concat(images, 0)

    return images


def noisetest(generator, step, mean_style, nrow, ncol):
    shape = 4 * 2 ** step
    alpha = 1

    images = []
    r = nrow - 1
    c = ncol - 1
    for i in range(nrow):
        source_code = jt.init.gauss((1, 512), 'float32')
        for k in range(ncol):
            noise = []
            for j in range(step+1):
                size = 4 * 2 ** j
                noise.append(jt.init.gauss((1, 1, size, size), "float32"))
            img_noise = generator(source_code, step=step, alpha=alpha,
                                   mean_style=mean_style, style_weight=0.7, noise=noise)
            images.append(img_noise)

    images = jt.concat(images, 0)

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')

    args = parser.parse_args()

    device = 'cuda'

    generator = StyledGenerator(512)
    #discriminator = Discriminator(from_rgb_activate=True)
    generator.load(args.path)
    #discriminator.load('./checkpoint/train_step-3-d.pkl')
    #generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator)

    step = int(math.log(args.size, 2)) - 2

    img = sample(generator, step, mean_style, args.n_row * args.n_col)
    jt.save_image(img, 'results/sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))

    #'''
    for j in range(2):
        img = style_mixing(generator, step, mean_style, args.n_col, args.n_row)
        jt.save_image(
            img, f'results/sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        )
    #'''

    ##row, col = 8, 10
    ##img = latentinterpolation(generator, step, mean_style, row, col)
    #'''
    img = latentinterpolation(generator, step, mean_style, args.n_row, args.n_col)
    jt.save_image(img, f'results/latentinter_{args.n_row}_{args.n_col}.png',
                  nrow=args.n_col, normalize=True, range=(-1, 1))
    #'''

    ##jt.save_image(img, f'results/latentinter_{row}_{col}.png',
    ##              nrow=col, normalize=True, range=(-1, 1)
    #noise test
    img = noisetest(generator, step, mean_style, args.n_row, args.n_col)
    jt.save_image(img, f'results/noisesample_{args.n_row}_{args.n_col}.png',
            nrow=args.n_col, normalize=True, range=(-1, 1))
