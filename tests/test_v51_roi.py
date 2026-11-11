import unittest
import numpy as np
from PIL import Image
from core.screen_guard import apply_guard, display_roi
class RoiTest(unittest.TestCase):
    def setup_img(self):
        a=np.full((300,500,3),170,np.uint8)
        a[35:270,40:460,:]=8
        a[115:155,150:210,:]=235 # bright oil inside black display
        return Image.fromarray(a)
    def test_guard_preserves_bright_oil_hole(self):
        img=self.setup_img();pred=np.zeros((300,500),np.uint8)
        pred[120:140,160:190]=1
        pred[5:12,5:10]=3
        pred[36:40,45:50]=3 # inside but within 16-pixel border
        fixed,info,mask=apply_guard(img,pred,16)
        self.assertEqual(info['status'],'applied')
        self.assertTrue(np.all(fixed[120:140,160:190]==1))
        self.assertEqual(int(fixed[5,5]),0)
        self.assertEqual(int(fixed[37,47]),0)
        self.assertTrue(mask[130,175])
    def test_fails_open_when_screen_unreliable(self):
        im=Image.fromarray(np.full((200,400,3),90,dtype=np.uint8));pred=np.ones((200,400),np.uint8)
        fixed,info,mask=apply_guard(im,pred)
        self.assertEqual(info['status'],'abstain')
        self.assertTrue((fixed==pred).all())
        self.assertIsNone(mask)
if __name__=='__main__':unittest.main()
