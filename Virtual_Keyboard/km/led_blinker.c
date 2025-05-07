#include <linux/module.h>      // Core header for Linux kernel modules
#include <linux/miscdevice.h>  // For miscellaneous character device support
#include <linux/fs.h>         // File system operations structure
#include <linux/uaccess.h>     // For user-space memory access functions

/**
 * Device write operation - handles data written to the device
 * 
 * @param f: Pointer to file structure
 * @param buf: User-space buffer containing data to write
 * @param len: Length of data to write
 * @param off: File offset pointer
 * @return: Number of bytes actually written or error code
 */
static ssize_t led_write(struct file *f, const char __user *buf,
                         size_t len, loff_t *off)
{
    char kbuf[2] = {0};  // Kernel buffer with null terminator
    
    // Only read first character if data exists
    if (len > 0) {
        // Safely copy 1 byte from user space to kernel space
        if (copy_from_user(kbuf, buf, 1))
            return -EFAULT;  // Return error if copy fails
            
        // Log received character to kernel log
        printk(KERN_INFO "led_blinker: Received character '%c'\n", kbuf[0]);
    }
    
    return len; // Return original length to indicate full write
}

/**
 * File operations structure - defines operations our device supports
 */
static const struct file_operations led_fops = {
    .owner = THIS_MODULE,  // Owner module to prevent unloading while in use
    .write = led_write,    // Our write operation handler
    // Note: read operation not implemented - device is write-only
};

/**
 * Miscellaneous device structure - describes our device
 */
static struct miscdevice led_dev = {
    .minor = MISC_DYNAMIC_MINOR,  // Let kernel assign minor number
    .name  = "led_blinker",       // Device name (appears in /dev)
    .fops  = &led_fops,           // Our file operations
    // Other fields (mode, nodename) use defaults
};

/**
 * Module initialization function
 * 
 * @return: 0 on success, error code on failure
 */
static int __init led_init(void)
{
    // Register our miscellaneous device
    int ret = misc_register(&led_dev);
    if (ret)
        // Log error if registration fails
        printk(KERN_ERR "led_blinker: Registration failed (%d)\n", ret);
    else
        // Log success message with device path
        printk(KERN_INFO "led_blinker: Ready at /dev/led_blinker\n");
    return ret;
}

/**
 * Module cleanup function
 */
static void __exit led_exit(void)
{
    // Unregister our device
    misc_deregister(&led_dev);
    // Log unregistration message
    printk(KERN_INFO "led_blinker: Unregistered\n");
}

// Specify module initialization and cleanup functions
module_init(led_init);
module_exit(led_exit);

// Module metadata
MODULE_LICENSE("GPL");  // GNU Public License v2 or later