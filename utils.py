from sklearn.linear_model import LinearRegression

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I

def infer(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        input = torch.from_numpy(image)
        input = input.permute((2,0,1))
        input = mytransform(input)
        image1 = input.permute((1,2,0)).numpy()
        input = input.to(torch.float32)
        input = input.unsqueeze(0)
        output = net(input)[0]
    output = output.permute((1,2,0)).numpy()
    mapping_func = get_mapping_func(image1, output)
    ret = outOfGamutClipping(apply_mapping_func(image, mapping_func))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)