import sys
import cv2

from ObstructionNetwork import ObstructionNetwork

GRAPH_PATH = "fence.pb"
IMGS = ["imgs/I0.png", "imgs/I1.png", "imgs/I2.png", "imgs/I3.png", "imgs/I4.png"]

def main():
    """
    Main execution function.
    """
    imgs = list(map(cv2.imread, IMGS))
    network = ObstructionNetwork(GRAPH_PATH)
    alpha, background = network.run(imgs)

    cv2.imwrite("alpha.png", alpha)
    cv2.imwrite("background.png", background)
    print("Wrote output images")
    sys.exit(0)


if __name__ == "__main__":
    main()
