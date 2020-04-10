from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    # ''' restricted boltzmann machine '''
    #
    # print ("\nStarting a Restricted Boltzmann Machine..")
    #
    # hidden = 500
    # losses = []
    # x = []
    #
    # while hidden >= 200:
    #     rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
    #                                      ndim_hidden=hidden,
    #                                      is_bottom=True,
    #                                      image_size=image_size,
    #                                      is_top=False,
    #                                      n_labels=10,
    #                                      batch_size=20)
    #
    #     res_loss = rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)
    #     losses.append(res_loss)
    #     x.append(hidden)
    #     print(losses)
    #     print(x)
    #     hidden -= 50
    #
    # plt.plot(x, losses)
    # plt.title("Trend of average reconstruction loss depending on number of hidden units")
    # plt.show()

    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
                        )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=30000)

    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    for digit in range(10):
        print("Gen digit n°%d" % digit)
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=30000)

    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    for digit in range(10):
        print("Gen digit n°%d" % digit)
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="dbn")
