import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple
import numpy as np
import laspy


def split(x_min, y_min, x_max, y_max, max_x_size, max_y_size, dim):
    bounds = []
    x_dim,y_dim = dim
    for x in range(x_dim):
        for y in range(y_dim):
            bounds.append((x_min+x*max_x_size, y_min+y*max_y_size, x_min+(x+1)*max_x_size, y_min+(y+1)*max_y_size))
    return bounds

def tuple_size(string):
    try:
        return tuple(map(float, string.split("x")))
    except:
        raise ValueError("Size must be in the form of numberxnumber eg: 50.0x65.14")

def splitter(input_file, output_dir, dim, itr_chunk=200_000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with laspy.open(input_file) as src:
        header = src.header
        nx, ny = dim

        if nx == 1 and ny == 1:
            with laspy.open(output_dir / "output_0.laz", mode="w", header=header) as dst:
                for points in src.chunk_iterator(itr_chunk):
                    dst.write_points(points)
            return

        x_size = (header.x_max - header.x_min) / nx 
        y_size = (header.y_max - header.y_min) / ny 
        sub_bounds = split(header.x_min, header.y_min, header.x_max, header.y_max, x_size, y_size, dim)

        writers: List[Optional[laspy.LasWriter]] = [None] * len(sub_bounds)

        try:
            for points in src.chunk_iterator(itr_chunk):

                xs = points.x
                ys = points.y

                for i, (bx0, by0, bx1, by1) in enumerate(sub_bounds):
                    mask = (xs >= bx0) & (xs < bx1) & (ys >= by0) & (ys < by1)

                    if not np.any(mask):
                        continue

                    if writers[i] is None:
                        out_path = output_dir / f"output_{i}.laz"
                        writers[i] = laspy.open(out_path, mode="w", header=header)

                    writers[i].write_points(points[mask])

                del mask, xs, ys, points

        finally:
            for w in writers:
                if w is not None:
                    w.close()

def main():
    parser = argparse.ArgumentParser(
        "LAS recursive splitter", description="Splits a las file bounds recursively"
    )
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("size", type=tuple_size, help="eg: 50x64.17")
    parser.add_argument("--points-per-iter", default=10**6, type=int)

    args = parser.parse_args()
    
    with laspy.open(sys.argv[1]) as file:
        max_x_size = (file.header.x_max - file.header.x_min) / args.size[0]
        max_y_size = (file.header.y_max - file.header.y_min) / args.size[1]
        sub_bounds = recursive_split(
            file.header.x_min,
            file.header.y_min,
            file.header.x_max,
            file.header.y_max,
            max_x_size,
            max_y_size,
        )

        writers: List[Optional[laspy.LasWriter]] = [None] * len(sub_bounds)
        pbar = tqdm(total=file.header.point_count)
        try:
            for points in file.chunk_iterator(args.points_per_iter):

                # For performance we need to use copy
                # so that the underlying arrays are contiguous
                x, y = points.x.copy(), points.y.copy()

                point_piped = 0

                for i, (x_min, y_min, x_max, y_max) in enumerate(sub_bounds):
                    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

                    if np.any(mask):
                        if writers[i] is None:
                            output_path = Path(sys.argv[2]) / f"output_{i}.laz"
                            writers[i] = laspy.open(
                                output_path, mode="w", header=file.header
                            )
                        sub_points = points[mask]
                        writers[i].write_points(sub_points)

                    point_piped += np.sum(mask)
                    if point_piped == len(points):
                        break
                pbar.update(len(points))
        finally:
            pbar.close()
            for writer in writers:
                if writer is not None:
                    writer.close()


if __name__ == "__main__":
    main()





