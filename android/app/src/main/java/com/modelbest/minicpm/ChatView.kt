package com.modelbest.minicpm

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
import android.provider.MediaStore
import android.text.TextUtils
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat.startActivityForResult
import androidx.navigation.NavController
import kotlinx.coroutines.launch
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.lang.Math.abs
import java.lang.Math.ceil
import java.lang.Math.log
import java.lang.Math.max
import java.lang.Math.min
import java.lang.Math.round
import java.lang.Math.sqrt


@ExperimentalMaterial3Api
@Composable
fun ChatView(
    navController: NavController, chatState: AppViewModel.ChatState, activity: Activity
) {
    val localFocusManager = LocalFocusManager.current
    (activity as MainActivity).chatState = chatState

    Scaffold(topBar = {
        TopAppBar(
            title = {
                Text(
                    text = "Demo: " + chatState.modelName.value, //.split("-")[0],
                    color = MaterialTheme.colorScheme.onPrimary
                )
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary),
            navigationIcon = {
                IconButton(
                    onClick = { navController.popBackStack() },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.ArrowBack,
                        contentDescription = "back home page",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            },
            actions = {
                IconButton(
                    onClick = {
                        chatState.requestResetChat()
                        activity.has_image = false
                    },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.Replay,
                        contentDescription = "reset the chat",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            })
    }, modifier = Modifier.pointerInput(Unit) {
        detectTapGestures(onTap = {
            localFocusManager.clearFocus()
        })
    }) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 10.dp)
        ) {
            val lazyColumnListState = rememberLazyListState()
            val coroutineScope = rememberCoroutineScope()
            Text(
                text = chatState.report.value,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight()
                    .padding(top = 5.dp)
            )
            Divider(thickness = 1.dp, modifier = Modifier.padding(vertical = 5.dp))
            LazyColumn(
                modifier = Modifier.weight(9f),
                verticalArrangement = Arrangement.spacedBy(5.dp, alignment = Alignment.Bottom),
                state = lazyColumnListState
            ) {
                coroutineScope.launch {
                    lazyColumnListState.animateScrollToItem(chatState.messages.size)
                }
                items(
                    items = chatState.messages,
                    key = { message -> message.id },
                ) { message ->
                    MessageView(messageData = message, activity)
                }
                item {
                    // place holder item for scrolling to the bottom
                }
            }
            Divider(thickness = 1.dp, modifier = Modifier.padding(top = 5.dp))
            SendMessageView(chatState = chatState, activity)
        }
    }
}
//对bitmap进行质量压缩
fun compressImage(image: Bitmap): Bitmap? {
    val baos = ByteArrayOutputStream()
    image.compress(Bitmap.CompressFormat.JPEG, 100, baos) //质量压缩方法，这里100表示不压缩，把压缩后的数据存放到baos中
    var options = 100
    while (baos.toByteArray().toString().length / 1024 > 100) {  //循环判断如果压缩后图片是否大于100kb,大于继续压缩
        baos.reset() //重置baos即清空baos
        image.compress(
            Bitmap.CompressFormat.JPEG,
            options,
            baos
        ) //这里压缩options%，把压缩后的数据存放到baos中
        options -= 10 //每次都减少10
    }
    val isBm =
        ByteArrayInputStream(baos.toByteArray()) //把压缩后的数据baos存放到ByteArrayInputStream中
    return BitmapFactory.decodeStream(isBm, null, null) //把ByteArrayInputStream数据生成图片
}
fun ResizeBitmap(image: Bitmap, newW : Int, newH: Int): Bitmap {
    return Bitmap.createScaledBitmap(image, newW, newH, true)
}

fun getImage(srcPath: String?): Bitmap? { //3 * 224 * 224
    if (TextUtils.isEmpty(srcPath)) //如果图片路径为空 直接返回
        return null
    val newOpts = BitmapFactory.Options()
    //开始读入图片，此时把options.inJustDecodeBounds 设回true了
    newOpts.inJustDecodeBounds = true
    var bitmap = BitmapFactory.decodeFile(srcPath, newOpts) //此时返回bm为空
    newOpts.inJustDecodeBounds = false
    val w = newOpts.outWidth
    val h = newOpts.outHeight
    //现在主流手机比较多是800*480分辨率，所以高和宽我们设置为
    val hh = 224f //这里设置高度为224f
    val ww = 224f //这里设置宽度为224f
    //缩放比。由于是固定比例缩放，只用高或者宽其中一个数据进行计算即可
    var be = 1 //be=1表示不缩放
    if (w > h && w > ww) { //如果宽度大的话根据宽度固定大小缩放
        be = (newOpts.outWidth / ww).toInt()
    } else if (w < h && h > hh) { //如果高度高的话根据宽度固定大小缩放
        be = (newOpts.outHeight / hh).toInt()
    }
    if (be <= 0) be = 1
    newOpts.inSampleSize = be //设置缩放比例
    //重新读入图片，注意此时已经把options.inJustDecodeBounds 设回false了
    bitmap = BitmapFactory.decodeFile(srcPath, newOpts)
    //return compressImage(bitmap) //压缩好比例大小后再进行质量压缩
    return ResizeBitmap(bitmap, 224, 224)
}

fun getOriginalImage(srcPath: String?): Bitmap? { //3 * 224 * 224
    return try {
        BitmapFactory.decodeFile(srcPath)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

fun bitmapToBytes(bitmap: Bitmap): IntArray {
    val width = bitmap.width
    val height = bitmap.height

    var pixels = IntArray(3 * height * width)

    for (y in 0 until height){
        for (x in 0 until width){
            val pixelColor = bitmap.getPixel(x, y) // 获取指定位置的像素值（ARGB格式）

            val redValue = Color.red(pixelColor) // 提取红色通道的值
            val greenValue = Color.green(pixelColor) // 提取绿色通道的值
            val blueValue = Color.blue(pixelColor) //
            pixels[0 * height * width + y * width + x] = redValue
            pixels[1 * height * width + y * width + x] = greenValue
            pixels[2 * height * width + y * width + x] = blueValue
        }
    }
    return pixels
}

fun ensure_divide(length: Int, patch_size: Int): Int {
    return max((round(length.toDouble() / patch_size) * patch_size).toInt(), patch_size)
}

data class ImageShape(val width: Int, val height: Int)
fun find_best_resize(
    original_size : IntArray ,
    scale_resolution: Int,
    patch_size: Int,
    allow_upscale : Boolean=false): ImageShape {
    var width = original_size[0]
    var height = original_size[1]
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale){
        val r = width.toDouble() / height
        height = (scale_resolution.toDouble() / sqrt(r)).toInt()
        width = (height * r).toInt()
    }

    var best_width = ensure_divide(width, patch_size)
    var best_height = ensure_divide(height, patch_size)
    return  ImageShape(best_width, best_height)
}

fun get_refine_size(
    original_size: IntArray,
    grid : IntArray,
    scale_resolution : Int,
    patch_size : Int,
    allow_upscale: Boolean=false
): ImageShape {
    var width = original_size[0]
    var height = original_size[1]
    var grid_x = grid[0]
    var grid_y = grid[1]

    var refine_width = ensure_divide(width, grid_x)
    var refine_height = ensure_divide(height, grid_y)

    var grid_width = refine_width / grid_x
    var grid_height = refine_height / grid_y

    var best_grid_size = find_best_resize(
        intArrayOf(grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale
    )

    var refine_size = ImageShape(best_grid_size.width * grid_x, best_grid_size.height * grid_y)

    return refine_size

}

/**
 * x和y是起始点
 * width和height是从起始点开始要裁剪的宽和高
 */
fun CropBitmap(source: Bitmap, x: Int, y: Int, width: Int, height: Int): Bitmap {
    // 创建一个新的Bitmap，其中包含裁剪后的图像
    val croppedBitmap = Bitmap.createBitmap(width, height, source.config)

    // 创建一个画布来绘制裁剪后的图像
    val canvas = android.graphics.Canvas(croppedBitmap)

    // 定义裁剪的区域
    val cropRect = Rect(x, y, x + width, y + height)

    // 使用画布绘制裁剪区域
    canvas.drawBitmap(source, cropRect, Rect(0, 0, width, height), null)

    return croppedBitmap
}
fun split_to_patches(image : Bitmap, grid: IntArray): Array<Bitmap> {
    var patches = arrayOf<Bitmap>()
    var width = image.width
    var height = image.height
    var grid_x = width / grid[0]
    var grid_y = height / grid[1]

    for (i in 0..height step grid_y){
        var images = arrayOf<Bitmap>()
        for (j in 0..width step grid_x){
            //var box = intArrayOf(j, i, j + grid_x, i + grid_y) //left, top, right, bottom
            //python: image.crop(box)
            var patch = CropBitmap(image, j, i, grid_x, grid_y)
            images += patch
        }
        patches += images
    }
    return patches

}

data class SliceResult(val image: Bitmap, val patchs: Array<Bitmap>, var best_grid: IntArray)
fun SliceImage(
    image: Bitmap,
    max_slice_nums: Int=9,
    scale_resolution: Int=448,
    patch_size: Int=14,
    never_split: Boolean=false): SliceResult {
    val original_height = image.height
    val original_width = image.width
    val original_size = intArrayOf(original_width, original_height)
    val log_ratio = log(original_width.toDouble() / original_height.toDouble())
    val ratio = (original_width * original_height).toDouble() / (scale_resolution * scale_resolution)
    val multiple = min(ceil(ratio).toInt(), max_slice_nums)
    var source_image : Bitmap
    var best_grid = intArrayOf(1, 1)

    var patches : Array<Bitmap> = arrayOf<Bitmap>()

    if (multiple <= 1 || never_split){
        var best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=false
        )
        //source_image = ResizeBitmap(image, best_size.width, best_size.height)
        source_image = ResizeBitmap(image, 224, 224)
    }else{
        var candidate_split_grids_nums : IntArray = intArrayOf()
        for (i in intArrayOf(multiple - 1, multiple, multiple + 1)) {
            if (i == 1 || i > max_slice_nums) {
                continue
            }
            candidate_split_grids_nums += i
        }
        //source image, down-sampling and ensure divided by patch_size
        var best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        //source_image = ResizeBitmap(image, best_resize.width, best_resize.height)
        source_image = ResizeBitmap(image, 224, 224)
        var candidate_grids = arrayOf<IntArray>()

        //find best grid
        for (split_grids_nums in candidate_split_grids_nums){
            var m = 1
            while (m <= split_grids_nums){
                if (split_grids_nums % m == 0){
                    candidate_grids += intArrayOf(m, split_grids_nums / m)
                }
                m += 1
            }
        }

        var min_error = Float.POSITIVE_INFINITY
        for (grid in candidate_grids) {
            var error = abs(log_ratio - log(grid[0].toDouble() / grid[1].toDouble()))
            if (error < min_error){
                best_grid = grid
                min_error = error.toFloat()
            }
        }

        var refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=true
        )

        //var refine_image = ResizeBitmap(image, refine_size.width, refine_size.height)
        var refine_image = ResizeBitmap(image, 224, 224)
        patches = split_to_patches(refine_image, best_grid)
    }
    return SliceResult(source_image, patches, best_grid)
}


@ExperimentalMaterial3Api
@Composable
fun MessageView(messageData: MessageData, activity: Activity) {
    var local_activity : MainActivity = activity as MainActivity
    SelectionContainer {
        if (messageData.role == MessageRole.Bot) {
            Row(
                horizontalArrangement = Arrangement.Start,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = messageData.text,
                    textAlign = TextAlign.Left,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier
                        .wrapContentWidth()
                        .background(
                            color = MaterialTheme.colorScheme.secondaryContainer,
                            shape = RoundedCornerShape(5.dp)
                        )
                        .padding(5.dp)
                        .widthIn(max = 300.dp)
                )
            }
        } else {
            Row(
                horizontalArrangement = Arrangement.End,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (messageData.image_path != "") {
                    var bitmap = getImage(messageData.image_path)
                    var original_bitmap = getOriginalImage(messageData.image_path)
                    if (bitmap != null) {
                        val image_data = bitmapToBytes(bitmap)
                        val slice_result = original_bitmap?.let { SliceImage(it,max_slice_nums=1) }
                        Log.v("get_image", image_data.size.toString())

                        Image(
                            bitmap.asImageBitmap(),
                            "",
                            modifier = Modifier
                                .wrapContentWidth()
                                .background(
                                    color = MaterialTheme.colorScheme.secondaryContainer,
                                    shape = RoundedCornerShape(5.dp)
                                )
                                .padding(5.dp)
                                .widthIn(max = 300.dp)
                        )
                        if (!local_activity.has_image) {
                            if (slice_result != null) {
                                var image_datas = arrayOf<IntArray>()
                                var image = slice_result.image
                                var image_data = bitmapToBytes(image)
                                image_datas += image_data
                                Log.v("image size = ", image.height.toString() + ", " + image.width.toString())

                                var steps : Int = 0
                                for (image in slice_result.patchs){
                                    var resize_image = ResizeBitmap(image, 224, 224)
                                    image_datas += bitmapToBytes(resize_image)
                                    Log.v("requestImage ", steps.toString())
                                }
                                var best_grid = slice_result.best_grid
                                local_activity.chatState.requestImage(image_datas, 224, 224, best_grid[0])
                                local_activity.slice_nums = slice_result.patchs.size
                            }
                            //local_activity.chatState.requestImage(image_data)
                        }
                        local_activity.has_image = true
                    }
                } else{
                    Text(
                        text = messageData.text,
                        textAlign = TextAlign.Left,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier
                            .wrapContentWidth()
                            .background(
                                color = MaterialTheme.colorScheme.primaryContainer,
                                shape = RoundedCornerShape(5.dp)
                            )
                            .padding(5.dp)
                            .widthIn(max = 300.dp)
                    )
                }
            }
        }
    }
}

@SuppressLint("UnrememberedMutableState")
@ExperimentalMaterial3Api
@Composable
fun SendMessageView(chatState: AppViewModel.ChatState, activity: Activity) {
    val localFocusManager = LocalFocusManager.current
    var local_activity : MainActivity = activity as MainActivity
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .height(IntrinsicSize.Max)
            .fillMaxWidth()
            .padding(bottom = 5.dp)
    ) {
        var text by rememberSaveable { mutableStateOf("") }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text(text = "Input") },
            modifier = Modifier
                .weight(9f),
        )
        if (chatState.modelName.value.endsWith("-V")) {
            IconButton(
                onClick = {
                    val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                    startActivityForResult(activity, intent, 1, null)
                    Log.v("get_image", "after startActivityForResult" + activity.image_path)
                },
                modifier = Modifier
                    .aspectRatio(1f)
                    .weight(1f),
                enabled = (chatState.chatable() && !local_activity.has_image)
            ) {
                Icon(
                    imageVector = Icons.Filled.AddAPhoto,
                    contentDescription = "use camera",
                )
            }
            IconButton(
                onClick = {
                    val intent = Intent()
                    intent.setType("image/*")
                    intent.setAction(Intent.ACTION_GET_CONTENT)
                    startActivityForResult(activity, Intent.createChooser(intent, "Select Picture"), 2, null)
                    Log.v("get_image", "after startActivityForResult" + activity.image_path)
                },
                modifier = Modifier
                    .aspectRatio(1f)
                    .weight(1f),
                enabled = (chatState.chatable() && !local_activity.has_image)
            ) {
                Icon(
                    imageVector = Icons.Filled.Photo,
                    contentDescription = "select image",
                )
            }
        }
        IconButton(
            onClick = {
                localFocusManager.clearFocus()
                chatState.requestGenerate(text, local_activity.slice_nums)
                text = ""
            },
            modifier = Modifier
                .aspectRatio(1f)
                .weight(1f),
            enabled = (text != "" && chatState.chatable() && (local_activity.has_image || !chatState.modelName.value.endsWith("-V")))
        ) {
            Icon(
                imageVector = Icons.Filled.Send,
                contentDescription = "send message",
            )
        }
    }
}
