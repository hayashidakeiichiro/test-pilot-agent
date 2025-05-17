from browser_use import Controller, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.dom.service import DomService
import logging
import asyncio
logger = logging.getLogger(__name__)


def attach_scroll_to_xpath(controller: Controller):
    @controller.action("Scroll the page to bring the element at the given xpath into view")
    async def scroll_to_xpath(xpath: str, browser: BrowserContext):  # type: ignore
        page = await browser.get_current_page()
        try:
            locator = page.locator(f'xpath={xpath}')
            if await locator.count() > 0:
                first = locator.first
                if await first.is_visible():
                    await first.scroll_into_view_if_needed()
                    await asyncio.sleep(0.5)  # Wait for scroll to complete
                    msg = f'🔍 Scrolled to XPath: {xpath}'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    msg = f"XPath '{xpath}' matched element, but it is not visible."
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                msg = f"No element found for XPath: '{xpath}'"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

        except Exception as e:
            msg = f"Failed to scroll to XPath '{xpath}': {str(e)}"
            logger.error(msg)
            return ActionResult(error=msg, include_in_memory=True)

