def parse_output(output: str) -> str:
    return output.split("ASSISTANT:")[-1].strip()

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _, _ in loader:
        batch_images_count = images.size(0)
        images = images.view(batch_images_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count

    mean /= total_images_count
    std /= total_images_count
    print(f"Mean: {mean}, Std: {std}, loader: {loader}")

    return mean, std