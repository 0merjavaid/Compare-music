import falcon
import os
from production.utils.utils import *
from training.model.backbone import Resnet
import requests
from PIL import Image
from torchvision import transforms
import torch
import time
import json
import torch.nn as nn


class Serve(object):

    def __init__(self, category_path, attributes_path):
        self.logger = get_logger(
            "production/loggings", "production.log", "serve", True)
        self.category, self.att_indices = parse_attributes(
            category_path, attributes_path)
        self.models = self.load_models(
            path="training/checkpoints/", gpu=True)
        self.normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])
        print ("Weighting for get req")

    def on_get(self, req, resp):
        s_time = time.time()
        self.logger.info(f'{req},{req.params}')
        p_resp = self.parse_req(req)
        if not p_resp[0]:
            resp.status = p_resp[1][0]
            resp.body = p_resp[1][1]
            return
        img_path = download_img("production/temp/", p_resp[0], 1)
        p_time = time.time()
        t_resp = self.img_to_tensor(img_path)
        if not t_resp[0]:
            resp.status = t_resp[1][0]
            resp.body = t_resp[1][1]
            return
        os.remove(img_path)
        pred = self.infer(t_resp[1], p_resp[1])
        save_img("production/temp/"+p_resp[0].split("/")
                 [-1], t_resp[0], p_resp[1], pred)

        e_time = time.time()
        print ("total", e_time-s_time, "Processing",
               e_time-p_time, "Downlaod", p_time-s_time)
        resp.body = json.dumps(pred)
        resp.status = falcon.HTTP_404

    def infer(self, tensor_img, category):
        attributes = self.category[category]
        resp = {}
        for attribute in attributes:
            attribute_names = self.att_indices[attribute]
            print (attribute,)
            if self.models[attribute] is not None:
                predictions = self.models[attribute](
                    tensor_img.view(1, 3, 224, 224).cuda())
                m = nn.Softmax()
                confidence = torch.max(m(predictions)).cpu().detach().numpy()
                pred_index = torch.max(predictions, 1)[1]
                resp[attribute] = [attribute_names[
                    pred_index], str(confidence*100)[:4]]

        return resp

    def img_to_tensor(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            image.load()
            tensor = self.preprocess(image)
            return image, tensor

        except Exception as e:
            print (e, "\n")
            status = falcon.HTTP_424
            error = e
            return False, (status, error)
        pass

    def parse_req(self, req):
        params = req.params
        status, error = None, None
        if len(params) == 0:
            status = falcon.HTTP_400
            error = "Zero Parameters found in the get request, Please add url and category in request"
            return False, (status, error)
        if "url" not in params.keys():
            print ("url not in params")
            status = falcon.HTTP_400
            error = "URL not found in the request, Please add url request"
            return False, (status, error)
            pass
        if "category" not in params.keys():
            status = falcon.HTTP_400
            error = "category not found in the request, Please add category request"
            return False, (status, error)
            print ("Cat not in params")

        if requests.head(params["url"]).status_code != requests.codes.ok:
            status = falcon.HTTP_404
            error = "Image URL tested, but failed to download image from provided URL"
            return False, (status, error)
            print("page wront")
        if params["category"].lower() not in self.category.keys():
            status = falcon.HTTP_404
            error = "Category not found in the trained networks"
            return False, (status, error)
        return params["url"], params["category"]

    def load_models(self, path, gpu):
        assert os.path.exists(path)
        models = {}
        attribute_weights = os.listdir(path)

        for attribute_weight in attribute_weights:
            pt_paths = os.listdir(os.path.join(path, attribute_weight))
            if (len(pt_paths) == 0):
                print (attribute_weight, "None")
                models[attribute_weight] = None
                continue

            best_val_path = sorted(pt_paths, key=lambda x: float(
                x.split("_")[-1].split(".")[1][: 3]))[-1]
            best_val_path = os.path.join(
                path, os.path.join(attribute_weight, best_val_path))
            assert os.path.exists(best_val_path)
            models[attribute_weight] = Resnet.get_model("resnet50", len(self.att_indices[attribute_weight]), False,
                                                        best_val_path, train=False)

            if gpu:
                models[attribute_weight] = models[attribute_weight].cuda()
            print ("Loaded ", attribute_weight)
        return models
