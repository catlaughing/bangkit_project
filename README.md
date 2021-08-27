# CaptionMe!
Bangkit Final Project
Project Name : CaptionMe!
By: Bandung 5 C
Description
Using CNN and LSTM to build an Automation Image Captioning. For image feature extraction, we use ResNet50 Architecture and ImageNet as pre-trained model. For processing the captions, we use GloVe Word Embedding.

This is step to step to run the app on your local environment:

1. After clone our repository, configure the flask app 
```
export FLASK_APP=app.py
```

2. Install the environment
```
pip install numpy
pip install tensorflow
pip install flask_gtts
```

3. Run our app!
```
flask run
```

4. Open your browser at:
```
https://127.0.0.1:5000/
```

Finish, you can try uploading your image!

Dataset Used: https://www.kaggle.com/shadabhussain/flickr8k
GloVe: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation

For more detailed instructions, you can check these notebooks:

Extracting image features: https://www.kaggle.com/intandea/image-captioning-extracting-features?scriptVersionId=35546590

Caption pre-processing and training model: https://www.kaggle.com/intandea/img-caption-train?scriptVersionId=36798939

Sample code for predicting captions: https://www.kaggle.com/intanyutami/test-my-photo?scriptVersionId=36987279

