import streamlit as st
import io

# Designing the interface
st.title("Deep Learning Steganography")

with st.spinner('Loading objects ...'):
    from model import *

tab1, tab2, tab3 = st.tabs(["About", "Hiding", "Recovery"])

tab1.subheader("About")

tab1.write('''Steganography is the practice of representing information within another message or physical object
            in such a manner that the presence of the information is not evident to human inspection. This app allows
            the user to do image-image steganography, which means that you can hide an image, called the "secret image",
            within another image, named the "cover image", without changing the visual appearance of the cover image.'''

            '''\n\nThere are two main components to this process: the hiding process and the revealing process. The hiding process
            refers to the merger of the cover and secret images, which yields a new image referred to as the "stego image" (short for steganographic image)
            which effectively contains both images but is "nearly identical" to the cover image. The second step is the revealing process, which takes as input the
            stego image and outputs an the "recovered secret image" that is "nearly identical" to the secret image. Here is an example of the full set of images:''')

with Image.open("example_comparison.png") as example_img:
    tab1.image(example_img)

tab1.write('''As we can see, there are slight differences between the cover and the stego image, such as color shifting of the background, a decrease in "sharpness",
            and other effects. However, someone that is not aware that this image went through a steganographic process will likely not be suspitious of this image in any way.
            The same could be said about the stego and recovered secret image, but given the nature of the images in this example, it is even harder to notice any differences.

            \n\nThere are many applications and use cases for which this process can be used, for example, this method can be used in the creation of nearly inperceptible watermarks and sending of secret messages.

            \n\n Watermarking: a photographer may want to use this tool to be able to publish one of his shots without a traditional watermark that is overlayed on top of the image
            which leads to a worst viewing experience for his fans. By using this tool, he can embed a watermark in the form of a secret image onto his picture (which will be the cover image)
            and this watermark will not be visible. However, if someone steals his property and distributes the photo it as if it was their own, the photographer can easily claim ownership
            of the photo by revealing the secret watermark embedded in the image. It is important to note that for this use case, the quality if of the cover image is much more important than
            the quality of the recovered secret image, as a watermark will be easily distinguishable even if heavily distorted but the cover image must still be as good as the original.

            \n\n Sending secret information: This process can also be used to store or send secret information in a way that cannot be easily detected. For example, imagine someone wants to save
            bank information on their phones but they know that taking a screenshot of your bank app is a liability because if the phone gets stolen your phone gallery may be compromised.
            This person may then hide this bank screenshot (as a secret image) within any other image they have in their phones, such as a photo of their families (the cover image), and it would
            probably be much safer this way than in its original form.''')

tab1.subheader("How to Use")

tab1.write('''On the top of this screen that are two tabs: "Hiding" and "Recovery", they are the two components required to achieve any of the goals mentioned above. These are the steps you must follow:
            \n 1. On the "Hiding" tab, Choose a cover and a secret image and upload them using the forms available. Remember that the cover will be kept the same and the secret will be hidden.
            \n 2. Click the "Run Model" button and wait for the stego image to be generated.
            \n 3. Download the stego image and save it locally (on your phone, computer, etc...) or send it to someone.
            \n 4. On the "Recovery" tab, upload the stego image on the upload form
            \n 5. Click the "Run Model" button and wait for the secret image to be recovered.
            \n It is important to note that on step 3, if you choose to store it locally, you must remember which cover image you picked and the URL of this website to retrieve it, if you forget any of these,
            it will be lost forever. Furthermore, if you send it to someone, this person must know that this is a stego image and the URL of this website to retrieve it, and remember,
            if you wish to send a secret message, it is recommended that you let this person know of these details beforehand. It is very conterproductive to send a message followed by
            the URL that is effectively the "key" to that secret message.''')

tab1.subheader("Limitations")

tab1.write('''The implementation available on this website have some limitations such as:
            \n 1. All the images uploaded and generated will have a 224x224 resolution, which is very small for most applications. New models will be created and uploaded to allow for greater resolution.
            \n 2. Only .jpg and .png files can be used, other image files will throw errors. Furthermore, all images will be converted to .png file type.
            \n 3. The model gives preference for better quality recovered secret images instead of better quality stego images. This means that the current model is not optimized for applications such as
            watermarking. In the future multiple models will be available and the user will be able to choose which image will be less distorted.''')


def get_image(img_type, tab_name, scale):
    bytes_data = None
    # Create image upload form and process image
    with tab_name.form(img_type + "form", clear_on_submit=True):
        uploaded_file = st.file_uploader(img_type + " Image")
        submitted = st.form_submit_button("Upload")
        if uploaded_file is not None:
            bytes_data = io.BytesIO(uploaded_file.getvalue())

    # Validate if image was uploaded
    if (bytes_data is None) and submitted:
        tab_name.write("No file was selected to upload")
        return 0, 0

    else:
        if bytes_data != None:
            # Open image and convert to RGB
            image = Image.open(bytes_data)
            metadata = image.text
            if image.mode != "RGB":
                image = image.convert(mode="RGB")

            # Rescale images so that they can be input into model
            width, height = scale, scale
            resized = image.resize(size=(width, height))
            return np.array(resized), metadata

        else:
            tab_name.markdown("Upload an Image.")
            return 0, 0

def display_image(img, where):
    show = where.image(img)

def reset_image(key):
    st.session_state[key] = 0

scale = 224
batch_size = 8

# Tab 2 - hiding

tab2.subheader("Hiding Process")
# Initialize state variables
if ('cover' not in st.session_state) and ('secret' not in st.session_state):
    st.session_state['cover'] = 0
    st.session_state['secret'] = 0

# Create image upload form for cover image
if isinstance(st.session_state['cover'], int):
    tab2.write('Pick a cover image.')
    cover, metadata = get_image('Cover', tab2, scale)
    if not isinstance(cover, int):
        st.session_state['cover'] = cover/255.
        st.session_state['cover_metadata'] = metadata

# Create button to repick cover image
if not isinstance(st.session_state['cover'], int):
    display_image(st.session_state['cover'], tab2)
    tab2.write('This is your cover image, the resulting steganographic image will be very similar to this.\n\nClick the button below to choose another image.')
    tab2.button("Repick Cover", on_click = reset_image, args = ('cover',), key="reset cover")

# Create image upload form for secret image
if isinstance(st.session_state['secret'], int):
    tab2.write('Pick a secret image.')
    secret, metadata = get_image('secret', tab2, scale)
    if not isinstance(secret, int):
        st.session_state['secret'] = secret/255.
        st.session_state['secret_metadata'] = metadata

# Create button to repick secret image
if not isinstance(st.session_state['secret'], int):
    display_image(st.session_state['secret'], tab2)
    tab2.write('This is your secret image, the resulting steganographic image will contain this image but it will be hidden.\n\nClick the button below to choose another image.')
    tab2.button("Repick Secret", on_click = reset_image, args = ('secret',), key="reset secret")

# Run Model button press definition
with st.spinner('Running Model'):
    if tab2.button('Run Model', 'Hide Button'):
        # Both images uploaded validation
        if isinstance(st.session_state['cover'], int) and isinstance(st.session_state['secret'], int):
            tab2.write("You must upload both images before running the model!")

        else:
            # Load model
            root = Path(os.getcwd())
            hide_path = root / 'hide'
            model = load_net(hide_path)

            # Run model
            stego_dwt = hide(model, st.session_state['secret'], st.session_state['cover'], batch_size, scale)

            # Do IDWT to obtain plain image for download
            input_tensor_idwt = tf.keras.layers.Input(shape = (int(scale/2), int(scale/2), 4, 3), dtype='float64', batch_size = 1)
            idwt_net = get_IDWT_TF(input_tensor_idwt, scale)
            idwt_stego = idwt_net(stego_dwt.reshape(1, int(scale/2), int(scale/2), 4, 3))
            stego_arr = np.reshape(idwt_stego, (scale, scale, 3))

            # Do normalization to be able to open image with PIL and show it
            norm_stego_arr = (stego_arr - np.min(stego_arr)) / (np.max(stego_arr) - np.min(stego_arr))
            stego_img = Image.fromarray((norm_stego_arr * 255).astype(np.uint8))
            show = tab2.image(stego_img, '\n\nStego Image')

            # Compute image metrics for nerds
            rmse = np.sqrt(np.mean((stego_arr - st.session_state['cover'])**2))
            psnr = calculate_psnr(stego_arr, st.session_state['cover'])
            ssim_val = ssim(stego_arr, st.session_state['cover'], multichannel = True)

            # Create Metadata containing normalization info required to do the inverse transforms
            final_metadata = PngInfo()
            for key, value in st.session_state['cover_metadata'].items():
                final_metadata.add_text(key, value)
            final_metadata.add_text("Norm", str(np.max(stego_arr)) + ',' +  str(np.min(stego_arr)))

            # Save image to buffer and open it
            buf = io.BytesIO()
            stego_img.save(buf, format="PNG", pnginfo=final_metadata)
            buf.seek(0)

            # Create download framework
            tab2.write('''This is your steganographic image, altough it looks very similar to the cover image, it is a two-in-one image, as it contains both the cover and the secret (hidden) on it.
                            You may download it using the button below and then use it for sending secret information, watermarking the cover image, etc... \n\nTo recover the secret image, just use this image
                            as the input on the "Recovery" tab seen above.''')
            tab2.download_button(
                label="Download Steganographic Image",
                data=buf.getvalue(),
                file_name='Downloaded Image.png',
                mime='image/png',
            )
            expander = tab2.expander("Extra information about the hiding process")
            expander.write('''Root Mean Squared Error = ''' + str(rmse)[:6] + ''' (Ideal Value is 0)\n\n Peak Signal to Noise Ration = ''' + str(psnr)[:5] +
                            ''' ("Acceptable" values are greater than 30) \n\n Structured Similarity Index Measure = ''' + str(ssim_val)[:5] + ''' (Ideal Value is 1)''')


#col1, col2, col3 , col4, col5 = st.beta_columns(5)

if ('stego' not in st.session_state):
    st.session_state['stego'] = 0

# Create image upload form for stego image
if isinstance(st.session_state['stego'], int):
    tab3.write('Pick a stego image.')
    stego, metadata = get_image('Stego', tab3, scale)
    if not isinstance(stego, int):
        st.session_state['stego'] = stego/255.
        st.session_state['metadata'] = metadata
        try:
            norm_max, norm_min = st.session_state['metadata']['Norm'].split(',')
            # Create button to repick stego image
        except:
            st.session_state['stego'] = 0
            tab3.error('It seems that the image you are trying to upload was not created using the "Hiding" tab or the image file is corrupted.')


if not isinstance(st.session_state['stego'], int):
    display_image(st.session_state['stego'], tab3)
    tab3.button("Repick Stego", on_click = reset_image, args = ('stego',), key="reset stego")
    tab3.write('This is your stego image, by running the model with the button below you will be able to retrieve the secret image hidden in it.')

with st.spinner('Running Model'):
    if tab3.button('Run Model', 'Reveal Button'):
        if isinstance(st.session_state['stego'], int):
            tab3.write("You must upload a stego image before running the model!")
        else:
            # Load Reveal Model
            root = Path(os.getcwd())
            reveal_path = root / 'reveal'
            model = load_net(reveal_path)

            # Retrieve normalization values from the image metadata
            norm_max, norm_min = st.session_state['metadata']['Norm'].split(',')
            norm_min = float(norm_min)
            norm_max = float(norm_max)

            # Do the inverse transformations, first the denormalization and then the DWT
            norm_stego = ((st.session_state['stego'])*(norm_max - norm_min)) + norm_min

            input_tensor_dwt = tf.keras.layers.Input(shape = (int(scale), int(scale), 3), dtype='float64', batch_size = 1)
            dwt_net = get_DWT_TF(input_tensor_dwt, scale)
            stego_dwt = dwt_net(np.reshape(norm_stego, (1, scale, scale, 3)))

            # Run the reveal model
            recovered_secret_arr = reveal(model, stego_dwt, batch_size, scale)
            # You may clip the values to be able to use the result as an image (yields better results)
            norm_recovered_secret_arr = np.clip(recovered_secret_arr, 0 , 1)

            # Or do normalization to be able to open image with PIL and show it
            # norm_recovered_secret_arr = (recovered_secret_arr - np.min(recovered_secret_arr)) / (np.max(recovered_secret_arr) - np.min(recovered_secret_arr))

            recovered_secret_img = Image.fromarray((norm_recovered_secret_arr * 255).astype(np.uint8))
            show = tab3.image(recovered_secret_img, '\n\nRecovered Secret')

            # Save image to buffer and open it
            buf = io.BytesIO()
            recovered_secret_img.save(buf, format="PNG")
            buf.seek(0)

            # Download Button
            tab3.download_button(
                label="Download stego image",
                data=buf.getvalue(),
                file_name='recovered secret image.png',
                mime='image/png',
            )
            tab3.write('''This is your recovered secret image, it likely looks very similar to the original secret image that was input into the hiding model.''')
